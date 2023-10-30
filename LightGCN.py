import scipy.sparse as sp
import torch.nn as nn
import numpy as np
import torch


class LightGCN(nn.Module):
    def __init__(self, data, n_users, n_items, n_layers, latent_dim, device):
        super(LightGCN, self).__init__()
        self.device = device
        self.data = data
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.init_embedding()
        self.norm_adj_mat_sparse_tensor = self.get_a_tilda()

    def init_embedding(self):
        self.E0 = nn.Embedding(self.n_users + self.n_items, self.latent_dim)
        nn.init.xavier_uniform_(self.E0.weight)
        self.E0.weight = nn.Parameter(self.E0.weight)

    def get_a_tilda(self):
        r = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        r[self.data['user_idx'], self.data['item_idx']] = 1.0

        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        r = r.tolil()

        adj_mat[: self.n_users, self.n_users:] = r
        adj_mat[self.n_users:, : self.n_users] = r.T
        adj_mat = adj_mat.todok()

        row_sum = np.array(adj_mat.sum(1))
        d_inv = np.power(row_sum + 1e-9, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_mat.dot(d_mat_inv)

        # Below Code is to convert the dok_matrix to sparse tensor.

        norm_adj_mat_coo = norm_adj_mat.tocoo().astype(np.float32)
        values = norm_adj_mat_coo.data
        indices = np.vstack((norm_adj_mat_coo.row, norm_adj_mat_coo.col))

        if self.device == 'cuda':
            i = torch.LongTensor(indices).cuda()
            v = torch.FloatTensor(values).cuda()
            shape = norm_adj_mat_coo.shape
            norm_adj_mat_sparse_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape)).cuda()
        else:
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = norm_adj_mat_coo.shape
            norm_adj_mat_sparse_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))

        return norm_adj_mat_sparse_tensor

    def propagate_through_layers(self):
        all_layer_embedding = [self.E0.weight]
        e_lyr = self.E0.weight

        if self.device == 'cuda':
            for layer in range(self.n_layers):
                e_lyr = torch.sparse.mm(self.norm_adj_mat_sparse_tensor, e_lyr).cuda()
                all_layer_embedding.append(e_lyr)
            all_layer_embedding = torch.stack(all_layer_embedding).cuda()
            mean_layer_embedding = torch.mean(all_layer_embedding, axis=0).cuda()
        else:
            for layer in range(self.n_layers):
                e_lyr = torch.sparse.mm(self.norm_adj_mat_sparse_tensor, e_lyr)
                all_layer_embedding.append(e_lyr)
            all_layer_embedding = torch.stack(all_layer_embedding)
            mean_layer_embedding = torch.mean(all_layer_embedding, axis=0)

        final_user_embed, final_item_embed = torch.split(mean_layer_embedding, [self.n_users, self.n_items])
        initial_user_embed, initial_item_embed = torch.split(self.E0.weight, [self.n_users, self.n_items])

        return final_user_embed, final_item_embed, initial_user_embed, initial_item_embed

    def forward(self, users, pos_items, neg_items):
        final_user_embed, final_item_embed, initial_user_embed, initial_item_embed = self.propagate_through_layers()

        users_emb, pos_emb, neg_emb = final_user_embed[users], final_item_embed[pos_items], final_item_embed[neg_items]
        user_emb0, pos_emb0, neg_emb0 = initial_user_embed[users], initial_item_embed[pos_items], initial_item_embed[
            neg_items]

        return users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0
