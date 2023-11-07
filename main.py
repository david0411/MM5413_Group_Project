import random
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pandas import DataFrame
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from LightGCN import LightGCN


def convert_to_sparse_tensor(dok_mtrx):
    dok_mtrx_coo = dok_mtrx.tocoo().astype(np.float32)
    values = dok_mtrx_coo.data
    indices = np.vstack((dok_mtrx_coo.row, dok_mtrx_coo.col))

    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    shape = dok_mtrx_coo.shape
    dok_mtrx_sparse_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)
    return dok_mtrx_sparse_tensor


def get_metrics(user_embed_wts, item_embed_wts, n_users, n_items, train_data, test_data, k):
    relevance_score = torch.matmul(user_embed_wts, torch.transpose(item_embed_wts, 0, 1)).to(device)

    r = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    r[train_data['user_idx'], train_data['item_idx']] = 1.0

    R_tensor = convert_to_sparse_tensor(r).to(device)
    R_tensor_dense = R_tensor.to_dense()
    R_tensor_dense = R_tensor_dense * (-np.inf)
    R_tensor_dense = torch.nan_to_num(R_tensor_dense, nan=0.0).to(device)

    relevance_score = relevance_score + R_tensor_dense

    top_k_relevance_indices = torch.topk(relevance_score, k).indices.to(device)

    top_k_relevance_indices_df = pd.DataFrame(top_k_relevance_indices.cpu().numpy(),
                                              columns=['top_indx_' + str(x + 1) for x in range(k)])

    top_k_relevance_indices_df['user'] = top_k_relevance_indices_df.index

    top_k_relevance_indices_df['top_rlvnt_itm'] = top_k_relevance_indices_df[
        ['top_indx_' + str(x + 1) for x in range(k)]].values.tolist()
    top_k_relevance_indices_df = top_k_relevance_indices_df[['user', 'top_rlvnt_itm']]

    test_interacted_items = test_data.groupby('user_idx')['item_idx'].apply(list).reset_index()

    metrics_df = pd.merge(test_interacted_items, top_k_relevance_indices_df, how='left', left_on='user_idx',
                          right_on=['user'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in
                                  zip(metrics_df.item_idx, metrics_df.top_rlvnt_itm)]

    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_idx']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / k, axis=1)

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


def bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):
    reg_loss = (1 / 2) * (user_emb0.norm(2).pow(2) +
                          pos_emb0.norm(2).pow(2) +
                          neg_emb0.norm(2).pow(2)) / float(len(users))
    pos_scores = torch.mul(users_emb, pos_emb).to(device)
    pos_scores = torch.sum(pos_scores, dim=1).to(device)
    neg_scores = torch.mul(users_emb, neg_emb).to(device)
    neg_scores = torch.sum(neg_scores, dim=1).to(device)
    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores)).to(device)

    return loss, reg_loss


def data_loader(data: DataFrame, batch_size, n_usr, n_itm):
    interacted_items_df = data.groupby('user_idx')['item_idx'].apply(list).reset_index()

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
    interacted_items_df = pd.merge(interacted_items_df, users_df, how='right', left_on='user_idx', right_on='users')
    pos_items = interacted_items_df['item_idx'].apply(lambda x: random.choice(x)).values
    neg_items = interacted_items_df['item_idx'].apply(lambda x: sample_neg(x)).values
    return list(users), list(pos_items), list(neg_items)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mse_loss(users, item_emb, itemEmb0):
    mse_loss = torch.sum(torch.square(torch.subtract(item_emb, itemEmb0))) / len(users)
    print(mse_loss)
    return mse_loss


if __name__ == '__main__':
    # recall_j_layer_set1 = []
    # recall_j_layer_set2 = []
    # recall_j_layer_set3 = []
    # for j in range(4):
    for i in range(3):
        mode = 'bpr'  # bpr or mse
        latent_dim = 64
        n_layers = 3
        EPOCHS = 300
        BATCH_SIZE = 1024
        DECAY = 1e-4
        K = 10
        seed_torch()
        loss_list_epoch_train = []
        mf_loss_list_epoch_train = []
        reg_loss_list_epoch_train = []
        loss_list_epoch_valid = []
        mf_loss_list_epoch_valid = []
        reg_loss_list_epoch_valid = []
        recall_list = []
        precision_list = []
        ndcg_list = []
        map_list = []
        train_time_list = []
        eval_time_list = []

        print('Dataset:{}'.format(i + 1))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(device))
        print('Parameters: Hidden Layers:{}, EPOCHS:{}, Decay:{}'.format(
            n_layers, EPOCHS, DECAY))

        pd.set_option('display.max_colwidth', None)
        columns_name = ['user', 'item', 'ratings']
        if i == 0:
            train_df = pd.read_csv("Data/Set1/train.txt")
            valid_df = pd.read_csv("Data/Set1/valid.txt")
            test_df = pd.read_csv("Data/Set1/test.txt")
        if i == 1:
            train_df = pd.read_csv("Data/Set2/ml_100k_full_set.csv")
            train_df, test_df = train_test_split(train_df.values, test_size=0.2, random_state=1)
            test_df, valid_df = train_test_split(test_df, test_size=0.5, random_state=1)
            train_df = pd.DataFrame(train_df, columns=columns_name)
            test_df = pd.DataFrame(test_df, columns=columns_name)
            valid_df = pd.DataFrame(valid_df, columns=columns_name)
        if i == 2:
            train_df = pd.read_csv("Data/Set3/imdb_100k_full_set.csv")
            train_df, test_df = train_test_split(train_df.values, test_size=0.2, random_state=1)
            test_df, valid_df = train_test_split(test_df, test_size=0.5, random_state=1)
            train_df = pd.DataFrame(train_df, columns=columns_name)
            test_df = pd.DataFrame(test_df, columns=columns_name)
            valid_df = pd.DataFrame(valid_df, columns=columns_name)
        le_user_train = pp.LabelEncoder()
        le_item_train = pp.LabelEncoder()
        le_user_valid = pp.LabelEncoder()
        le_item_valid = pp.LabelEncoder()

        train_df['user_idx'] = le_user_train.fit_transform(train_df['user'].values)
        train_df['item_idx'] = le_item_train.fit_transform(train_df['item'].values)
        train_user = train_df['user'].unique()
        train_item = train_df['item'].unique()
        n_users_train = train_df['user_idx'].nunique()
        n_items_train = train_df['item_idx'].nunique()
        print("Number of Unique Users:", n_users_train)
        print("Number of unique Items:", n_items_train)

        test_df = test_df[(test_df['user'].isin(train_user)) & (test_df['item'].isin(train_item))]
        test_df['user_idx'] = le_user_train.transform(test_df['user'].values)
        test_df['item_idx'] = le_item_train.transform(test_df['item'].values)

        valid_df = valid_df[(valid_df['user'].isin(train_user)) & (valid_df['item'].isin(train_item))]
        valid_df['user_idx'] = le_user_valid.fit_transform(valid_df['user'].values)
        valid_df['item_idx'] = le_item_valid.fit_transform(valid_df['item'].values)
        n_users_valid = valid_df['user_idx'].nunique()
        n_items_valid = valid_df['item_idx'].nunique()

        lightGCN = LightGCN(train_df, n_users_train, n_items_train, n_layers, latent_dim, device).to(device)
        print("Size of Learnable Embedding:", list(lightGCN.parameters())[0].size())

        optimizer = torch.optim.Adam(lightGCN.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

        for epoch in tqdm(range(EPOCHS)):
            n_batch = int(len(train_df) / BATCH_SIZE)

            final_loss_list_train = []
            mf_loss_list_train = []
            reg_loss_list_train = []
            final_loss_list_valid = []
            mf_loss_list_valid = []
            reg_loss_list_valid = []

            best_ndcg = -1

            train_start_time = time.time()
            lightGCN.train()
            for batch_idx in range(n_batch):
                optimizer.zero_grad()

                users_train, pos_items_train, neg_items_train = (
                    data_loader(train_df, BATCH_SIZE, n_users_train, n_items_train))
                if mode == 'bpr':
                    users_emb_train, pos_emb_train, neg_emb_train, userEmb0_train, posEmb0_train, negEmb0_train = (
                        lightGCN.forward(users_train, pos_items_train, neg_items_train, mode))

                    mf_loss_train, reg_loss_train = bpr_loss(
                        users_train,
                        users_emb_train,
                        pos_emb_train,
                        neg_emb_train,
                        userEmb0_train,
                        posEmb0_train,
                        negEmb0_train)

                    reg_loss_train = DECAY * reg_loss_train
                    final_loss_train = mf_loss_train + reg_loss_train

                    final_loss_train.backward()
                    optimizer.step()

                    final_loss_list_train.append(final_loss_train.item())
                    mf_loss_list_train.append(mf_loss_train.item())
                    reg_loss_list_train.append(reg_loss_train.item())
                else:
                    users_emb_train, item_emb_train, userEmb0_train, itemEmb0_train = (
                        lightGCN.forward(users_train, pos_items_train, neg_items_train, mode))

                    loss_train = mse_loss(users_train, item_emb_train, itemEmb0_train)

                    loss_train.backward()
                    optimizer.step()
                    final_loss_list_train.append(loss_train.item())

            # scheduler.step()
            train_end_time = time.time()
            train_time = train_end_time - train_start_time
            lightGCN.eval()
            with ((torch.no_grad())):
                final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = (
                    lightGCN.propagate_through_layers())

                users_valid, pos_items_valid, neg_items_valid = (
                    data_loader(valid_df, BATCH_SIZE, n_users_valid, n_items_valid))
                if mode == 'bpr':
                    users_emb_valid, pos_emb_valid, neg_emb_valid, userEmb0_valid, posEmb0_valid, negEmb0_valid = (
                        lightGCN.forward(users_valid, pos_items_valid, neg_items_valid, mode))

                    mf_loss_valid, reg_loss_valid = bpr_loss(
                        users_valid,
                        users_emb_valid,
                        pos_emb_valid,
                        neg_emb_valid,
                        userEmb0_valid,
                        posEmb0_valid,
                        negEmb0_valid)

                    reg_loss_valid = DECAY * reg_loss_valid
                    final_loss_valid = mf_loss_valid + reg_loss_valid

                    final_loss_list_valid.append(final_loss_valid.item())
                    mf_loss_list_valid.append(mf_loss_valid.item())
                    reg_loss_list_valid.append(reg_loss_valid.item())
                else:
                    users_emb_valid, item_emb_valid, userEmb0_valid, itemEmb0_valid, = (
                        lightGCN.forward(users_valid, pos_items_valid, neg_items_valid, mode))

                    loss_valid = mse_loss(users_valid, item_emb_valid, itemEmb0_valid)

                    final_loss_list_valid.append(loss_valid.item())

                test_topK_recall, test_topK_precision, test_topK_ndcg, test_topK_map = get_metrics(final_user_Embed,
                                                                                                   final_item_Embed,
                                                                                                   n_users_train,
                                                                                                   n_items_train,
                                                                                                   train_df,
                                                                                                   test_df,
                                                                                                   K)

            if test_topK_ndcg > best_ndcg:
                best_ndcg = test_topK_ndcg
                torch.save(final_user_Embed, 'final_user_Embed.pt')
                torch.save(final_item_Embed, 'final_item_Embed.pt')
                torch.save(initial_user_Embed, 'initial_user_Embed.pt')
                torch.save(initial_item_Embed, 'initial_item_Embed.pt')

            eval_time = time.time() - train_end_time

            loss_list_epoch_train.append(round(np.mean(final_loss_list_train), 4))
            mf_loss_list_epoch_train.append(round(np.mean(mf_loss_list_train), 4))
            reg_loss_list_epoch_train.append(round(np.mean(reg_loss_list_train), 4))

            loss_list_epoch_valid.append(round(np.mean(final_loss_list_valid), 4))
            mf_loss_list_epoch_valid.append(round(np.mean(mf_loss_list_valid), 4))
            reg_loss_list_epoch_valid.append(round(np.mean(reg_loss_list_valid), 4))

            recall_list.append(round(test_topK_recall, 4))
            precision_list.append(round(test_topK_precision, 4))
            ndcg_list.append(round(test_topK_ndcg, 4))
            map_list.append(round(test_topK_map, 4))
            train_time_list.append(train_time)
            eval_time_list.append(eval_time)

        epoch_list = [(i + 1) for i in range(EPOCHS)]

        # Accuracy data
        # plt.plot(epoch_list, recall_list, label='Recall')
        # plt.plot(epoch_list, precision_list, label='Precision')
        # plt.plot(epoch_list, ndcg_list, label='NDCG')
        # plt.plot(epoch_list, map_list, label='MAP')
        # plt.xlabel('Epoch')
        # plt.ylabel('Metrics')
        # plt.legend()
        # plt.title("Accuracy")
        # plt.show()

        # Training Loss plot
        # plt.plot(epoch_list, loss_list_epoch_train, label='Total Training Loss')
        # plt.plot(epoch_list, mf_loss_list_epoch_train, label='MF Training Loss')
        # plt.plot(epoch_list, reg_loss_list_epoch_train, label='Reg Training Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.title("Training Loss")
        # plt.show()

        # Validation Loss plot
        # plt.plot(epoch_list, loss_list_epoch_valid, label='Total Validating Loss')
        # plt.plot(epoch_list, mf_loss_list_epoch_valid, label='MF Validating Loss')
        # plt.plot(epoch_list, reg_loss_list_epoch_valid, label='Reg Validating Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.title("Validation Loss")
        # plt.show()
        print(loss_list_epoch_train)
        print(loss_list_epoch_valid)

        # Training and Validation Loss plot
        plt.plot(epoch_list, loss_list_epoch_train, label='Total Training Loss')
        plt.plot(epoch_list, loss_list_epoch_valid, label='Total Validating Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.show()

        print("Average time taken to train an epoch -> ", round(np.mean(train_time_list), 2), " seconds")
        print("Average time taken to eval an epoch -> ", round(np.mean(eval_time_list), 2), " seconds")
        print("Test Data Recall -> ", recall_list[-1])
        print("Test Data Precision -> ", precision_list[-1])
        print("Test Data NDCG -> ", ndcg_list[-1])
        print("Test Data MAP -> ", map_list[-1])
        print("Train Data Loss -> ", loss_list_epoch_train[-1])
        print("Valid Data Loss -> ", loss_list_epoch_valid[-1])
        print()

        # if i == 0:
        #     recall_j_layer_set1.append(recall_list[-1])
        # if i == 1:
        #     recall_j_layer_set2.append(recall_list[-1])
        # if i == 2:
        #     recall_j_layer_set3.append(recall_list[-1])

# Recall for different layers
# plt.plot([0, 1, 2, 3], recall_j_layer_set1, label='Recall')
# plt.plot([0, 1, 2, 3], recall_j_layer_set2, label='Recall')
# plt.plot([0, 1, 2, 3], recall_j_layer_set3, label='Recall')
# plt.xlabel('Layers')
# plt.ylabel('Recall')
# plt.legend()
# plt.title('Recall by layer')
# plt.show()
