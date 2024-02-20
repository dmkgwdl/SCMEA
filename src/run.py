#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import gc
from pprint import pprint

import scipy
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.distance import braycurtis
from torch_geometric.utils import sort_edge_index
from lion_optim import *
from models import *
from utils import *
from loss import *
from load import *


def train():
    print("[start training...] ")


class SCMEA:
    def __init__(self):

        self.nhop_path = None
        self.rel_ht_dict = None
        self.parallel_co_attention = None
        self.nca_loss = None
        self.ent_name_dim = None
        self.ent_features = None
        self.triples_2 = None
        self.triples_1 = None
        self.line_graph_index_in = None
        self.line_graph_index_out = None
        self.ent2id_dict = None
        self.ills = None
        self.triples = None
        self.r_hs = None
        self.r_ts = None
        self.ids = None
        self.left_ents = None
        self.right_ents = None
        self.long_tail_en = []
        self.edge_index_all = None
        self.rel_all = None
        self.rel_features_in = None
        self.rel_features_out = None
        self.att_features = None

        self.left_non_train = None
        self.right_non_train = None
        self.ENT_NUM = None
        self.REL_NUM = None
        self.e_adj = None
        self.r_adj = None
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
        self.sim_module = None

        self.gcn_pro = None
        self.rel_pro = None
        self.attr_pro = None
        self.img_pro = None

        self.input_dim = None
        self.entity_emb = None
        self.e_input_idx = None
        self.r_input_idx = None
        self.n_units = None
        self.n_heads = None
        self.cross_graph_model = None
        self.params = None
        self.optimizer = None

        self.loss = None
        self.cur_max_epoch = 0

        self.multi_loss_layer = None
        self.align_multi_loss_layer = None
        self.fusion = None  # fusion module

        self.parser = argparse.ArgumentParser()
        self.args = self.parse_options(self.parser)

        self.set_seed(self.args.seed, self.args.cuda)

        self.device = torch.device("cuda" if self.args.cuda and torch.cuda.is_available() else "cpu")

        self.init_data()
        self.init_emb()
        self.init_model()

    @staticmethod
    def parse_options(parser):
        parser.add_argument("--file_dir", type=str, default="data/DBP15K/fr_en", required=False,
                            help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
        parser.add_argument("--rate", type=float, default=0.3, help="training set rate")
        parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
        parser.add_argument("--seed", type=int, default=2023, help="random seed")
        parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
        parser.add_argument("--check_point", type=int, default=100, help="check point")
        parser.add_argument("--check_point_il", type=int, default=50, help="check point after il")
        parser.add_argument("--hidden_units", type=str, default="128,128,128",
                            help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
        parser.add_argument("--rel_hidden_units", type=str, default="100,100,100",
                            help="hidden units in each rel hidden layer(including in_dim and out_dim)")
        parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
        parser.add_argument("--rel_heads", type=str, default="1", help="heads in each gat layer, splitted with comma")
        parser.add_argument("--instance_normalization", action="store_true", default=False,
                            help="enable instance normalization")
        parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
        parser.add_argument("--weight_decay", type=float, default=0, help="weight decay (L2 loss on parameters)")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
        parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
        parser.add_argument("--dist", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')")
        parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference")
        parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
        parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
        parser.add_argument("--semi_learn_step", type=int, default=10, help="If IL, what's the update step?")
        parser.add_argument("--il_start", type=int, default=500, help="If Il, when to start?")
        parser.add_argument("--bsize", type=int, default=7500, help="batch size")
        parser.add_argument("--unsup", action="store_true", default=False)
        parser.add_argument("--lta_split", type=int, default=0, help="split in {0,1,2,3,|splits|-1}")
        parser.add_argument("--with_weight", type=int, default=1, help="Whether to weight the fusion of different "
                                                                       "modal features")
        parser.add_argument("--structure_encoder", type=str, default="gcn", help="the encoder of structure view, "
                                                                                 "[gcn|gat]")
        parser.add_argument("--rel_structure_encoder", type=str, default="gcn", help="the encoder of relation view, "
                                                                                     "[gcn|gat]")
        parser.add_argument("--w_triple_gat", type=int, default=1, help="Whether to use the W-Triple GAT")
        parser.add_argument("--optimizer", type=str, default="AdamW", help="AdamW | Lion")
        parser.add_argument("--cl", action="store_true", default=True, help="CL")
        parser.add_argument("--tau", type=float, default=0.1, help="the temperature factor of contrastive loss")
        parser.add_argument("--ab_weight", type=float, default=0.5, help="the weight of NTXent Loss")
        parser.add_argument("--attr_dim", type=int, default=100, help="the hidden size of attr and rel features")
        parser.add_argument("--rel_dim", type=int, default=100, help="the hidden size of relation feature")
        parser.add_argument("--w_gph", action="store_false", default=True, help="with gph features")
        parser.add_argument("--w_rel", action="store_false", default=True, help="with rel features")
        parser.add_argument("--w_attr", action="store_false", default=True, help="with attr features")
        parser.add_argument("--w_in_gph", type=int, default=0, help="interactive_learning with gph features")
        parser.add_argument("--w_in_rel", type=int, default=0, help="interactive_learning with rel features")
        parser.add_argument("--w_in_att", type=int, default=0, help="interactive_learning with att features")
        parser.add_argument("--expend_t", type=float, default=0.3, help="expend entities before training")
        parser.add_argument("--pro_lte", type=float, default=0.3, help="The proportion of long tail entities")
        parser.add_argument("--w_lg", type=int, default=0, help="with lg features")
        parser.add_argument("--w_ra", type=int, default=0, help="with ra?")
        parser.add_argument("--inner_view_num", type=int, default=4, help="the number of inner view")
        parser.add_argument("--loss", type=str, default="nca", help="[nca|hinge]")
        parser.add_argument("--gamma", type=float, default=3, help="expend entities before training")
        parser.add_argument("--word_embedding", type=str, default="wc", help="the type of name embedding, "
                                                                                "[glove|wc]")
        parser.add_argument("--ent_name", type=int, default=0, help="init with entity name")
        parser.add_argument("--use_project_head", action="store_true", default=True, help="use projection head")

        parser.add_argument("--zoom", type=float, default=0.1, help="narrow the range of losses")
        parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]")
        parser.add_argument("--save_path", type=str, default="save", help="save path")
        return parser.parse_args()

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
        file_dir = self.args.file_dir
        device = self.device

        self.ent2id_dict, self.ills, self.triples_1, self.triples_2, self.triples, self.ids, self.rel_ht_dict = read_raw_data(
            file_dir, kg_list)
        e1 = os.path.join(file_dir, 'ent_ids_1')
        e2 = os.path.join(file_dir, 'ent_ids_2')
        self.left_ents = get_ids(e1)
        self.right_ents = get_ids(e2)
        self.ENT_NUM = len(self.ent2id_dict)
        self.REL_NUM = len(self.rel_ht_dict)
        np.random.shuffle(self.ills)
        self.train_ill = np.array(self.ills[:int(len(self.ills) // 1 * self.args.rate)], dtype=np.int32)
        self.test_ill_ = self.ills[int(len(self.ills) // 1 * self.args.rate):]
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

        self.left_non_train = list(set(self.left_ents) - set(self.train_ill[:, 0].tolist()))
        self.right_non_train = list(set(self.right_ents) - set(self.train_ill[:, 1].tolist()))

        print("-----dataset summary-----")
        print("dataset:\t", file_dir)
        print("triple num:\t", len(self.triples))
        print("entity num:\t", self.ENT_NUM)
        print("relation num:\t", self.REL_NUM)
        print("#left entity : %d, #right entity: %d" % (len(self.left_ents), len(self.right_ents)))
        print("train ill num:\t", self.train_ill.shape[0], "\ttest ill num:\t", self.test_ill.shape[0])
        print("-------------------------")

    def init_emb(self):
        file_dir = self.args.file_dir
        device = self.device
        if self.args.word_embedding == "glove" and self.args.ent_name > 0:
            if "SRPRS" in file_dir:
                vec_path = file_dir + '/' + file_dir.split('/')[-1] + '_word.npy'
                with open(vec_path, 'rb') as f:
                    ent_features = np.load(f)
                    ent_features = torch.Tensor(ent_features)
            else:
                word2vec_path = file_dir + '/' + file_dir.split('/')[-1].split('_')[0] + '_vectorList.json'
                with open(word2vec_path, 'r', encoding='utf-8') as f:
                    ent_features = torch.tensor(json.load(f))
            ent_features.requires_grad = True
            self.ent_features = torch.Tensor(ent_features).to(device)
            self.ent_name_dim = self.ent_features.shape[1]
        if self.args.word_embedding == "wc" and self.args.ent_name > 0:
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

    def init_model(self):
        rel_size = self.rel_features_in.shape[1]
        attr_size = self.att_features.shape[1]

        self.multiview_encoder = MultiViewEncoder(args=self.args, device=self.device,
                                                  ent_num=self.ENT_NUM,
                                                  rel_num=self.REL_NUM,
                                                  name_size=self.ent_name_dim,
                                                  rel_size=rel_size, attr_size=attr_size,
                                                  use_project_head=False).to(self.device)
        
        # self.parallel_co_attention = Parallel_Co_Attention(hidden_dim=300)
        self.params = [
            {"params":
                 list(self.multiview_encoder.parameters())
                 # list(self.parallel_co_attention.parameters())
             }]

        if self.args.loss == "nca":
            # multi-view loss
            print("using NCA loss")
            self.gcn_nca_loss = NCA_loss(alpha=5, beta=10, ep=0.0, device=self.device)
            self.nca_loss = NCA_loss(alpha=15, beta=10, ep=0.0, device=self.device)
            if self.args.cl:
                print("CL!")
                self.loss = BiCl(device=self.device, tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        else:
            print("using Hinge loss")
            # self.loss = L1_Loss(self.args.gamma)

        # select optimizer
        if self.args.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.params,
                lr=self.args.lr
            )
        else:
            self.optimizer = Lion(
                self.params,
                lr=self.args.lr
            )

        print("--------------------model details--------------------")
        print("SCMEA model details:")
        print(self.multiview_encoder)
        print("optimiser details:")
        print(self.optimizer)

    def semi_supervised_learning(self, epoch):
        with torch.no_grad():
            gph_emb, ent_rel_emb, rel_emb, att_emb, joint_emb, tri_emb, name_emb = self.multiview_encoder(self.e_input_idx,
                                                                                                self.r_input_idx,
                                                                                                self.e_in, self.e_out,
                                                                                                self.e_adj, epoch,
                                                                                                self.r_in_adj,
                                                                                                self.r_out_adj,
                                                                                                self.r_path_adj,
                                                                                                self.edge_index_all,
                                                                                                self.rel_all,
                                                                                                self.ent_features,
                                                                                                self.rel_features_in,
                                                                                                self.rel_features_out,
                                                                                                self.att_features)

            if tri_emb is not None:
                tri_emb = F.normalize(tri_emb)
            
            joint_emb = F.normalize(joint_emb)

        distance_list = []
        d_f = None
        for i in np.arange(0, len(self.left_non_train), 1000):
            d_joi = pairwise_distances(joint_emb[self.left_non_train[i:i + 1000]], joint_emb[self.right_non_train])
            if tri_emb is not None:
                d_tri = pairwise_distances(tri_emb[self.left_non_train[i:i + 1000]], tri_emb[self.right_non_train])
                d_f = d_joi + d_tri
            else:
                d_f = d_joi
            distance_list.append(d_f)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance
        del gph_emb, ent_rel_emb, rel_emb, att_emb, joint_emb, tri_emb, name_emb
        return preds_l, preds_r

    # Train
    def train(self):
        args = self.args
        pprint(args)
        print("[start training...] ")
        new_links = []
        max_hit1 = .0
        epoch_CG = 0
        bsize = self.args.bsize
        device = self.device

        self.e_input_idx = torch.LongTensor(np.arange(self.ENT_NUM)).to(device)
        self.r_input_idx = torch.LongTensor(np.arange(self.REL_NUM)).to(device)
        for epoch in range(self.args.epochs):
            if self.args.optimizer == "AdamW" and epoch == epoch >= self.args.il_start:
                self.optimizer = optim.AdamW(self.params, lr=self.args.lr / 5)

            t_epoch = time.time()
            self.multiview_encoder.train()

            self.optimizer.zero_grad()

            gph_emb, ent_rel_emb, rel_emb, att_emb, joint_emb, tri_emb, name_emb = self.multiview_encoder(self.e_input_idx,
                                                                                                self.r_input_idx,
                                                                                                self.e_in, self.e_out,
                                                                                                self.e_adj, epoch,
                                                                                                self.r_in_adj,
                                                                                                self.r_out_adj,
                                                                                                self.r_path_adj,
                                                                                                self.edge_index_all,
                                                                                                self.rel_all,
                                                                                                self.ent_features,
                                                                                                self.rel_features_in,
                                                                                                self.rel_features_out,
                                                                                                self.att_features)

            loss_sum_all, loss_tri_all, loss_attr_all, loss_joi_all = 0, 0, 0, 0
            epoch_CG += 1

            np.random.shuffle(self.train_ill)
            for si in np.arange(0, self.train_ill.shape[0], args.bsize):
                if tri_emb is not None:
                    if self.args.cl:
                        icl_loss = self.loss(tri_emb, self.train_ill[si:si + bsize])
                    else:
                        icl_loss = self.gcn_nca_loss(tri_emb, self.train_ill[si:si + bsize], [], device=self.device)
                else:
                    icl_loss = self.loss(joint_emb, self.train_ill[si:si + bsize])
                loss_all = self.gcn_nca_loss(joint_emb, self.train_ill[si:si + bsize], [], device=self.device)

                loss_all = loss_all + icl_loss
                loss_sum_all = loss_sum_all + loss_all

            loss_sum_all.backward()
            self.optimizer.step()

            print("[epoch {:d}] loss_all: {:f}, time: {:.4f} s".format(epoch, loss_sum_all.item(), time.time() - t_epoch))
            del gph_emb, rel_emb, att_emb, joint_emb, tri_emb, ent_rel_emb, name_emb

            if epoch >= self.args.il_start and (epoch + 1) % self.args.semi_learn_step == 0 and self.args.il:
                pred_left, pred_right = self.semi_supervised_learning(epoch)

                if (epoch + 1) % (self.args.semi_learn_step * 10) == self.args.semi_learn_step:
                    new_links = [(self.left_non_train[i], self.right_non_train[p]) for i, p in enumerate(pred_left)
                                 if pred_right[p] == i]
                else:
                    new_links = [(self.left_non_train[i], self.right_non_train[p]) for i, p in enumerate(pred_left)
                                 if (pred_right[p] == i)
                                 and ((self.left_non_train[i], self.right_non_train[p]) in new_links)]

            if epoch >= self.args.il_start and (epoch + 1) % (self.args.semi_learn_step * 10) == 0 and len(
                    new_links) != 0 and self.args.il:
                new_links_elect = new_links
                self.train_ill = np.vstack((self.train_ill, np.array(new_links_elect)))
                num_true = len([nl for nl in new_links_elect if nl in self.test_ill_])
                for nl in new_links_elect:
                    self.left_non_train.remove(nl[0])
                    self.right_non_train.remove(nl[1])

                new_links = []

            if self.args.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Test
            if epoch < self.args.il_start and (epoch + 1) % self.args.check_point == 0:
                print("\n[epoch {:d}] checkpoint!".format(epoch))
                max_hit1, max_epoch = self.test(epoch, max_hit1)
                if max_epoch > self.cur_max_epoch:
                    self.cur_max_epoch = max_epoch
            if epoch >= self.args.il_start and (epoch + 1) % self.args.check_point_il == 0:
                print("\n[epoch {:d}] checkpoint!".format(epoch))
                max_hit1, max_epoch = self.test(epoch, max_hit1)
                if max_epoch > self.cur_max_epoch:
                    self.cur_max_epoch = max_epoch
            if self.args.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("max_hit1 = {}, cur_max_epoch = {}".format(max_hit1, self.cur_max_epoch))
        print("[optimization finished!]")
        # torch.save(self.multiview_encoder, os.path.join(self.args.save_path, "model_myEA.pth"))
        print("[save the EA model finished!]")

    # Test
    def test(self, epoch, max_hit1):
        print("\n[epoch {:d}] checkpoint!".format(epoch))
        with torch.no_grad():
            t_test = time.time()
            self.multiview_encoder.eval()

            gph_emb, ent_rel_emb, rel_emb, att_emb, joint_emb, tri_emb, name_emb = self.multiview_encoder(self.e_input_idx,
                                                                                                self.r_input_idx,
                                                                                                self.e_in, self.e_out,
                                                                                                self.e_adj, epoch,
                                                                                                self.r_in_adj,
                                                                                                self.r_out_adj,
                                                                                                self.r_path_adj,
                                                                                                self.edge_index_all,
                                                                                                self.rel_all,
                                                                                                self.ent_features,
                                                                                                self.rel_features_in,
                                                                                                self.rel_features_out,
                                                                                                self.att_features)

            w_normalized = F.softmax(self.multiview_encoder.fusion.weight, dim=0)
            print("normalised weights:", w_normalized.data.squeeze())
            if tri_emb is not None:
                tri_emb = F.normalize(tri_emb)
            
            joint_emb = F.normalize(joint_emb)

            # top_k = [1, 5, 10, 50, 100]
            top_k = [1, 5, 10, 50]
            if "100" in self.args.file_dir:
                pass
            else:
                acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
                acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
                test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
                if self.args.dist == 2:
                    distance_joi = pairwise_distances(joint_emb[self.test_left], joint_emb[self.test_right])
                    if tri_emb is not None:
                        distance_tri = pairwise_distances(tri_emb[self.test_left], tri_emb[self.test_right])
                        distance_f = distance_joi + distance_tri
                    else:
                        distance_f = distance_joi
                elif self.args.dist == 1:
                    distance_joi = torch.FloatTensor(spatial.distance.cdist(
                        joint_emb[self.test_left].cpu().data.numpy(),
                        joint_emb[self.test_right].cpu().data.numpy(), metric="cityblock"))
                    if tri_emb is not None:
                        distance_tri = torch.FloatTensor(spatial.distance.cdist(
                            tri_emb[self.test_left].cpu().data.numpy(),
                            tri_emb[self.test_right].cpu().data.numpy(), metric="cityblock"))
                        distance_f = distance_joi + distance_tri
                    else:
                        distance_f = distance_joi
                else:
                    raise NotImplementedError
                distance = distance_f

                if self.args.csls is True:
                    distance = 1 - csls_sim(1 - distance, self.args.csls_k)

                if epoch + 1 == self.args.epochs:
                    to_write = []
                    test_left_np = self.test_left.cpu().numpy()
                    test_right_np = self.test_right.cpu().numpy()
                    to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])

                for idx in range(self.test_left.shape[0]):
                    values, indices = torch.sort(distance[idx, :], descending=False)
                    rank = (indices == idx).nonzero().squeeze().item()

                    mean_l2r += (rank + 1)
                    mrr_l2r += 1.0 / (rank + 1)
                    for i in range(len(top_k)):
                        if rank < top_k[i]:
                            acc_l2r[i] += 1
                    if epoch + 1 == self.args.epochs:
                        indices = indices.cpu().numpy()
                        to_write.append(
                            [idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]],
                             test_right_np[indices[1]], test_right_np[indices[2]]])
                if epoch + 1 == self.args.epochs:
                    import csv
                    save_path = self.args.save_path
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    with open(os.path.join(save_path, "pred.txt"), "w") as f:
                        wr = csv.writer(f, dialect='excel')
                        wr.writerows(to_write)

                for idx in range(self.test_right.shape[0]):
                    _, indices = torch.sort(distance[:, idx], descending=False)
                    rank = (indices == idx).nonzero().squeeze().item()
                    mean_r2l += (rank + 1)
                    mrr_r2l += 1.0 / (rank + 1)
                    for i in range(len(top_k)):
                        if rank < top_k[i]:
                            acc_r2l[i] += 1

                mean_l2r /= self.test_left.size(0)
                mean_r2l /= self.test_right.size(0)
                mrr_l2r /= self.test_left.size(0)
                mrr_r2l /= self.test_right.size(0)
                for i in range(len(top_k)):
                    acc_l2r[i] = round(acc_l2r[i] / self.test_left.size(0), 4)
                    acc_r2l[i] = round(acc_r2l[i] / self.test_right.size(0), 4)

                del gph_emb, rel_emb, att_emb, joint_emb, tri_emb, ent_rel_emb, name_emb
                gc.collect()

            avg_hit = (acc_l2r + acc_r2l) / 2
            print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r,
                                                                                                mean_l2r, mrr_l2r,
                                                                                                time.time() - t_test))
            print("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_r2l,
                                                                                                  mean_r2l, mrr_r2l,
                                                                                                  time.time() - t_test))
            print("avg_hit = {}\tavg_mr = {:.3f}\tavg_mrr={:.3f}\n".format(avg_hit, (mean_l2r+mean_r2l)/2, (mrr_l2r+mrr_r2l)/2))
            if avg_hit[0] > max_hit1:
                return avg_hit[0], epoch
            else:
                return max_hit1, self.cur_max_epoch


if __name__ == "__main__":
    model = SCMEA()
    model.train()
