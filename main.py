import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn import preprocessing as pp
from tqdm.notebook import tqdm

pd.set_option('display.max_colwidth', None)
columns_name = ['user', 'item', 'ratings']
train_df = pd.read_csv("./Data/train.txt")
valid_df = pd.read_csv("./Data/valid.txt")
test_df = pd.read_csv("./Data/test.txt")

le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()
train_df['user_idx'] = le_user.fit_transform(train_df['user'].values)
train_df['item_idx'] = le_item.fit_transform(train_df['item'].values)
train_user = train_df['user'].unique()
train_item = train_df['item'].unique()

test_df = test_df[(test_df['user'].isin(train_user)) & (test_df['item'].isin(train_item))]
valid_df = valid_df[(valid_df['user'].isin(train_user)) & (valid_df['item'].isin(train_item))]

valid_df['user_idx'] = le_user.transform(valid_df['user'].values)
valid_df['item_idx'] = le_item.transform(valid_df['item'].values)
test_df['user_idx'] = le_user.transform(test_df['user'].values)
test_df['item_idx'] = le_item.transform(test_df['item'].values)

n_users = train_df['user_idx'].nunique()
n_items = train_df['item_idx'].nunique()
print("Number of Unique Users : ", n_users)
print("Number of unique Items : ", n_items)

latent_dim = 64
n_layers = 3


def convert_to_sparse_tensor(dok_mtrx):
    dok_mtrx_coo = dok_mtrx.tocoo().astype(np.float32)
    values = dok_mtrx_coo.data
    indices = np.vstack((dok_mtrx_coo.row, dok_mtrx_coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = dok_mtrx_coo.shape

    dok_mtrx_sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return dok_mtrx_sparse_tensor


def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_data, test_data, K):
    user_Embedding = nn.Embedding(user_Embed_wts.size()[0], user_Embed_wts.size()[1], _weight=user_Embed_wts)
    item_Embedding = nn.Embedding(item_Embed_wts.size()[0], item_Embed_wts.size()[1], _weight=item_Embed_wts)

    test_user_ids = torch.LongTensor(test_data['user_idx'].unique())

    relevance_score = torch.matmul(user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1))

    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    R[train_data['user_idx'], train_data['item_idx']] = 1.0

    R_tensor = convert_to_sparse_tensor(R)
    R_tensor_dense = R_tensor.to_dense()

    R_tensor_dense = R_tensor_dense * (-np.inf)
    R_tensor_dense = torch.nan_to_num(R_tensor_dense, nan=0.0)

    relevance_score = relevance_score + R_tensor_dense

    topk_relevance_score = torch.topk(relevance_score, K).values
    topk_relevance_indices = torch.topk(relevance_score, K).indices

    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.numpy(),
                                             columns=['top_indx_' + str(x + 1) for x in range(K)])

    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index

    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[
        ['top_indx_' + str(x + 1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user', 'top_rlvnt_itm']]

    test_interacted_items = test_data.groupby('user_idx')['item_idx'].apply(list).reset_index()

    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how='left', left_on='user_idx',
                          right_on=['user'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in
                                  zip(metrics_df.item_idx, metrics_df.top_rlvnt_itm)]

    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_idx']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)

    def get_hit_list(item_idx, top_rlvnt_itm):
        return [1 if x in set(item_idx) else 0 for x in top_rlvnt_itm]

    metrics_df['hit_list'] = metrics_df.apply(lambda x: get_hit_list(x['item_idx'], x['top_rlvnt_itm']), axis=1)

    def get_dcg_idcg(item_idx, hit_list):
        idcg = sum([1 / np.log1p(idx + 1) for idx in range(min(len(item_idx), len(hit_list)))])
        dcg = sum([hit / np.log1p(idx + 1) for idx, hit in enumerate(hit_list)])
        return dcg / idcg

    def get_cumsum(hit_list):
        return np.cumsum(hit_list)

    def get_map(item_idx, hit_list, hit_list_cumsum):
        return sum([hit_cumsum * hit / (idx + 1) for idx, (hit, hit_cumsum) in
                    enumerate(zip(hit_list, hit_list_cumsum))]) / len(item_idx)

    metrics_df['ndcg'] = metrics_df.apply(lambda x: get_dcg_idcg(x['item_idx'], x['hit_list']), axis=1)
    metrics_df['hit_list_cumsum'] = metrics_df.apply(lambda x: get_cumsum(x['hit_list']), axis=1)

    metrics_df['map'] = metrics_df.apply(lambda x: get_map(x['item_idx'], x['hit_list'], x['hit_list_cumsum']),
                                         axis=1)

    return metrics_df['recall'].mean(), metrics_df['precision'].mean(), metrics_df['ndcg'].mean(), metrics_df[
        'map'].mean()


class LightGCN(nn.Module):
    def __init__(self, data, n_users, n_items, n_layers, latent_dim):
        super(LightGCN, self).__init__()
        self.data = data
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.init_embedding()
        self.norm_adj_mat_sparse_tensor = self.get_A_tilda()

    def init_embedding(self):
        self.E0 = nn.Embedding(self.n_users + self.n_items, self.latent_dim)
        nn.init.xavier_uniform_(self.E0.weight)
        self.E0.weight = nn.Parameter(self.E0.weight)

    def get_A_tilda(self):
        R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        R[self.data['user_idx'], self.data['item_idx']] = 1.0

        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = R.tolil()

        adj_mat[: n_users, n_users:] = R
        adj_mat[n_users:, : n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_mat.dot(d_mat_inv)

        # Below Code is toconvert the dok_matrix to sparse tensor.

        norm_adj_mat_coo = norm_adj_mat.tocoo().astype(np.float32)
        values = norm_adj_mat_coo.data
        indices = np.vstack((norm_adj_mat_coo.row, norm_adj_mat_coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = norm_adj_mat_coo.shape

        norm_adj_mat_sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        return norm_adj_mat_sparse_tensor

    def propagate_through_layers(self):
        all_layer_embedding = [self.E0.weight]
        E_lyr = self.E0.weight

        for layer in range(self.n_layers):
            E_lyr = torch.sparse.mm(self.norm_adj_mat_sparse_tensor, E_lyr)
            all_layer_embedding.append(E_lyr)

        all_layer_embedding = torch.stack(all_layer_embedding)
        mean_layer_embedding = torch.mean(all_layer_embedding, axis=0)

        final_user_Embed, final_item_Embed = torch.split(mean_layer_embedding, [n_users, n_items])
        initial_user_Embed, initial_item_Embed = torch.split(self.E0.weight, [n_users, n_items])

        return final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed

    def forward(self, users, pos_items, neg_items):
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = self.propagate_through_layers()

        users_emb, pos_emb, neg_emb = final_user_Embed[users], final_item_Embed[pos_items], final_item_Embed[neg_items]
        userEmb0, posEmb0, negEmb0 = initial_user_Embed[users], initial_item_Embed[pos_items], initial_item_Embed[
            neg_items]

        return users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0


lightGCN = LightGCN(train_df, n_users, n_items, n_layers, latent_dim)
print("Size of Learnable Embedding : ", list(lightGCN.parameters())[0].size())


def bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0):
    reg_loss = (1 / 2) * (userEmb0.norm().pow(2) +
                          posEmb0.norm().pow(2) +
                          negEmb0.norm().pow(2)) / float(len(users))
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)

    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

    return loss, reg_loss


def data_loader(data, batch_size, n_usr, n_itm):
    interected_items_df = data.groupby('user_idx')['item_idx'].apply(list).reset_index()

    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    indices = [x for x in range(n_usr)]

    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)

    users.sort()

    users_df = pd.DataFrame(users, columns=['users'])

    interected_items_df = pd.merge(interected_items_df, users_df, how='right', left_on='user_idx', right_on='users')

    pos_items = interected_items_df['item_idx'].apply(lambda x: random.choice(x)).values

    neg_items = interected_items_df['item_idx'].apply(lambda x: sample_neg(x)).values

    return list(users), list(pos_items), list(neg_items)


optimizer = torch.optim.Adam(lightGCN.parameters(), lr=0.005)

EPOCHS = 30
BATCH_SIZE = 1024
DECAY = 0.0001
K = 10

loss_list_epoch = []
MF_loss_list_epoch = []
reg_loss_list_epoch = []

recall_list = []
precision_list = []
ndcg_list = []
map_list = []

train_time_list = []
eval_time_list = []

for epoch in tqdm(range(EPOCHS)):
    n_batch = int(len(train_df) / BATCH_SIZE)

    final_loss_list = []
    MF_loss_list = []
    reg_loss_list = []

    best_ndcg = -1

    train_start_time = time.time()
    lightGCN.train()
    for batch_idx in range(n_batch):
        optimizer.zero_grad()
        users, pos_items, neg_items = data_loader(train_df, BATCH_SIZE, n_users, n_items)
        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = lightGCN.forward(users, pos_items, neg_items)
        mf_loss, reg_loss = bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0)
        reg_loss = DECAY * reg_loss
        final_loss = mf_loss + reg_loss
        final_loss.backward()
        optimizer.step()

        final_loss_list.append(final_loss.item())
        MF_loss_list.append(mf_loss.item())
        reg_loss_list.append(reg_loss.item())

    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    lightGCN.eval()
    with (torch.no_grad()):
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = lightGCN.propagate_through_layers()
        test_topK_recall, test_topK_precision, test_topK_ndcg, test_topK_map = get_metrics(final_user_Embed, final_item_Embed, n_users,n_items, train_df, test_df, K)

    if test_topK_ndcg > best_ndcg:
        best_ndcg = test_topK_ndcg
        torch.save(final_user_Embed, 'final_user_Embed.pt')
        torch.save(final_item_Embed, 'final_item_Embed.pt')
        torch.save(initial_user_Embed, 'initial_user_Embed.pt')
        torch.save(initial_item_Embed, 'initial_item_Embed.pt')

    eval_time = time.time() - train_end_time

    loss_list_epoch.append(round(np.mean(final_loss_list), 4))
    MF_loss_list_epoch.append(round(np.mean(MF_loss_list), 4))
    reg_loss_list_epoch.append(round(np.mean(reg_loss_list), 4))
    recall_list.append(round(test_topK_recall, 4))
    precision_list.append(round(test_topK_precision, 4))
    ndcg_list.append(round(test_topK_ndcg, 4))
    map_list.append(round(test_topK_map, 4))
    train_time_list.append(train_time)
    eval_time_list.append(eval_time)

epoch_list = [(i+1) for i in range(EPOCHS)]

plt.plot(epoch_list, recall_list, label='Recall')
plt.plot(epoch_list, precision_list, label='Precision')
plt.plot(epoch_list, ndcg_list, label='NDCG')
plt.plot(epoch_list, map_list, label='MAP')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()

plt.plot(epoch_list, loss_list_epoch, label='Total Training Loss')
plt.plot(epoch_list, MF_loss_list_epoch, label='MF Training Loss')
plt.plot(epoch_list, reg_loss_list_epoch, label='Reg Training Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

print("Average time taken to train an epoch -> ", round(np.mean(train_time_list),2), " seconds")
print("Average time taken to eval an epoch -> ", round(np.mean(eval_time_list),2), " seconds")

print("Last Epoch's Test Data Recall -> ", recall_list[-1])
print("Last Epoch's Test Data Precision -> ", precision_list[-1])
print("Last Epoch's Test Data NDCG -> ", ndcg_list[-1])
print("Last Epoch's Test Data MAP -> ", map_list[-1])

print("Last Epoch's Train Data Loss -> ", loss_list_epoch[-1])