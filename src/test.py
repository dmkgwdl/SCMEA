#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
from tqdm import tqdm
import os
from torch_geometric.utils import sort_edge_index
from lion_optim import *
from models import *
from utils import *
from loss import *
from load import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TestModel:
    def __init__(self, dataset):
        self.ent_name_dim = None
        self.ent_features = None
        self.triples_2 = None
        self.triples_1 = None
        self.ent2id_dict = None
        self.ills = None
        self.triples = None
        self.ids = None
        self.left_ents = None
        self.right_ents = None
        self.edge_index_all = None
        self.rel_all = None
        self.rel_features_in = None
        self.rel_features_out = None
        self.att_features = None
        self.ENT_NUM = None
        self.REL_NUM = None
        self.e_adj = None
        self.r_in_adj = None
        self.r_out_adj = None
        self.r_path_adj = None
        self.train_ill = None
        self.test_ill_ = None
        self.test_ill = None
        self.test_left = None
        self.test_right = None
        self.e_in = None
        self.e_out = None
        self.multiview_encoder = None
        self.e_input_idx = None
        self.r_input_idx = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_dir = dataset
        self.set_seed(2023, cuda=True if torch.cuda.is_available() else "cpu")

        self.init_data()
        self.init_emb()

        self.e_input_idx = torch.LongTensor(np.arange(self.ENT_NUM)).to(self.device)
        self.r_input_idx = torch.LongTensor(np.arange(self.REL_NUM)).to(self.device)

    @staticmethod
    def set_seed(seed, cuda=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def init_data(self):
        # Load data
        kg_list = [1, 2]
        file_dir = self.file_dir
        device = self.device

        self.ent2id_dict, self.ills, self.triples_1, self.triples_2, self.triples, self.ids, self.rel_ht_dict = read_raw_data(
            file_dir, kg_list)
        e1 = os.path.join(file_dir, 'ent_ids_1')
        e2 = os.path.join(file_dir, 'ent_ids_2')
        self.left_ents = get_ids(e1)
        self.right_ents = get_ids(e2)
        self.ENT_NUM = len(self.ent2id_dict)
        self.REL_NUM = len(self.rel_ht_dict)
        self.test_ill_ = self.ills[int(len(self.ills) // 1 * 0.3):]
        self.test_ill = np.array(self.test_ill_, dtype=np.int32)
        head_l = []
        rel_l = []
        tail_l = []
        for (head, rel, tail) in self.triples:
            head_l.append(head)
            rel_l.append(rel)
            tail_l.append(tail)
        head_l = torch.tensor(head_l, dtype=torch.long)
        rel_l = torch.tensor(rel_l, dtype=torch.long)
        # print(rel_l.max())
        tail_l = torch.tensor(tail_l, dtype=torch.long)

        edge_index = torch.stack([head_l, tail_l], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel_l)
        edge_index_all = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        rel_all = torch.cat([rel, rel + rel.max() + 1])
        self.edge_index_all = edge_index_all.to(device)
        self.rel_all = rel_all.to(device)

        self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).to(device)
        self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).to(device)

    def init_emb(self):
        file_dir = self.file_dir
        device = self.device
        vec_path = file_dir + '/' + file_dir.split('/')[-1] + '_wc.npy'
        with open(vec_path, 'rb') as f:
            ent_features = np.load(f)
        self.ent_features = torch.Tensor(ent_features).to(device)
        self.ent_features.requires_grad = True
        self.ent_name_dim = self.ent_features.shape[1]
        a1 = os.path.join(file_dir, 'training_attrs_1')
        a2 = os.path.join(file_dir, 'training_attrs_2')
        self.att_features = load_attr([a1, a2], self.ENT_NUM, self.ent2id_dict, 1000)
        self.att_features = torch.Tensor(self.att_features).to(device)
        self.rel_features_in, self.rel_features_out = load_relation(self.ENT_NUM, self.REL_NUM, self.triples)
        self.rel_features_in = torch.Tensor(self.rel_features_in).to(device)
        self.rel_features_out = torch.Tensor(self.rel_features_out).to(device)
        self.e_adj = get_adjr(self.ENT_NUM, self.triples, norm=True)  # getting a sparse tensor r_adj
        self.e_adj = self.e_adj.to(self.device)


dataset = "../data/DBP15K/zh_en"
sub_kg = dataset.split('/')[-1]

model_path = os.path.join("../save_pkl", sub_kg, sub_kg + "_SCMEA.pth")

net = TestModel(dataset)

net.multiview_encoder = torch.load(model_path,
                                   map_location=torch.device('cuda') if torch.cuda.is_available() else "cpu")

save_path = os.path.join("../test_logs/", sub_kg)

if not os.path.exists(save_path):
    os.mkdir(save_path)

with torch.no_grad():
    print("[Start testing...] ")
    t_test = time.time()
    net.multiview_encoder.eval()
    _, _, _, _, joint_emb, tri_emb, _ = net.multiview_encoder(net.e_input_idx, net.r_input_idx, net.e_in, net.e_out,
                                                              net.e_adj, -1, net.r_in_adj, net.r_out_adj,
                                                              net.r_path_adj, net.edge_index_all, net.rel_all,
                                                              net.ent_features,
                                                              net.rel_features_in, net.rel_features_out,
                                                              net.att_features)

    tri_emb = F.normalize(tri_emb)
    joint_emb = F.normalize(joint_emb)

    top_k = [1, 5, 10, 50]
    acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
    acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
    test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
    distance_joi = pairwise_distances(joint_emb[net.test_left], joint_emb[net.test_right])
    distance_tri = pairwise_distances(tri_emb[net.test_left], tri_emb[net.test_right])
    distance_f = distance_joi + distance_tri
    distance = 1 - csls_sim(1 - distance_f, 10)

    to_write = []
    test_left_np = net.test_left.cpu().numpy()
    test_right_np = net.test_right.cpu().numpy()
    to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])

    print("[Testing from left to right...] ")
    for idx in tqdm(range(net.test_left.shape[0])):
        values, indices = torch.sort(distance[idx, :], descending=False)
        rank = (indices == idx).nonzero().squeeze().item()

        mean_l2r += (rank + 1)
        mrr_l2r += 1.0 / (rank + 1)
        for i in range(len(top_k)):
            if rank < top_k[i]:
                acc_l2r[i] += 1
        indices = indices.cpu().numpy()
        to_write.append(
            [idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]],
             test_right_np[indices[1]], test_right_np[indices[2]]])

    with open(os.path.join(save_path, "pred.txt"), "w") as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerows(to_write)

    print("[Testing from right to left...] ")
    for idx in tqdm(range(net.test_right.shape[0])):
        _, indices = torch.sort(distance[:, idx], descending=False)
        rank = (indices == idx).nonzero().squeeze().item()
        mean_r2l += (rank + 1)
        mrr_r2l += 1.0 / (rank + 1)
        for i in range(len(top_k)):
            if rank < top_k[i]:
                acc_r2l[i] += 1

    mean_l2r /= net.test_left.size(0)
    mean_r2l /= net.test_right.size(0)
    mrr_l2r /= net.test_left.size(0)
    mrr_r2l /= net.test_right.size(0)
    for i in range(len(top_k)):
        acc_l2r[i] = round(acc_l2r[i] / net.test_left.size(0), 4)
        acc_r2l[i] = round(acc_r2l[i] / net.test_right.size(0), 4)

    avg_hit = (acc_l2r + acc_r2l) / 2
    print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r,
                                                                                        mean_l2r, mrr_l2r,
                                                                                        time.time() - t_test))
    print("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_r2l,
                                                                                        mean_r2l, mrr_r2l,
                                                                                        time.time() - t_test))
    print("avg_hit = {}\tavg_mr = {:.3f}\tavg_mrr={:.3f}\n".format(avg_hit, (mean_l2r + mean_r2l) / 2,
                                                                   (mrr_l2r + mrr_r2l) / 2))
