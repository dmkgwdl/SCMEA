#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_sparse import spmm

from layers import *

from compgcn_conv import CompGCNConv


class Parallel_Co_Attention(nn.Module):
    def __init__(self, hidden_dim, k=300, dropout=0.2, alpha=0.7, device=None):
        super(Parallel_Co_Attention, self).__init__()
        self.alpha = alpha
        self.W_b = nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim)))
        nn.init.xavier_uniform_(self.W_b.data, gain=1.414)
        self.W_b.requires_grad = True

        self.W_l = nn.Parameter(torch.zeros(size=(k, hidden_dim)))
        nn.init.xavier_uniform_(self.W_l.data, gain=1.414)
        self.W_l.requires_grad = True
        self.W_r = nn.Parameter(torch.zeros(size=(k, hidden_dim)))
        nn.init.xavier_uniform_(self.W_r.data, gain=1.414)
        self.W_r.requires_grad = True
        self.w_hl = nn.Parameter(torch.zeros(size=(1, k)))
        nn.init.xavier_uniform_(self.W_r.data, gain=1.414)
        self.w_hl.requires_grad = True
        self.w_hr = nn.Parameter(torch.zeros(size=(1, k)))
        nn.init.xavier_uniform_(self.W_r.data, gain=1.414)
        self.w_hr.requires_grad = True

        self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()
        print("using Parallel")

    # left_emb:(N * d), right_emb:(T * d)
    def forward(self, left_emb, right_emb):
        # affinity_matrix = self.tanh(torch.mm(right_emb, torch.mm(self.W_b, left_emb.t())))  # T * N
        # left_trans_att = F.softmax(affinity_matrix.t(), dim=1)  # N * T
        # right_trans_att = F.softmax(affinity_matrix, dim=1)     # T * N
        # print(left_trans_att.shape, right_emb.shape)
        N = left_emb.shape[0]
        T = right_emb.shape[0]
        left_emb_T = left_emb.t()
        right_emb_T = right_emb.t()
        affinity_matrix = torch.tanh(torch.matmul(right_emb, torch.matmul(self.W_b, left_emb.t())))  # T * N

        h_left = torch.matmul(self.W_l, left_emb_T) + torch.matmul(torch.matmul(self.W_r, right_emb_T), affinity_matrix)
        h_left = self.dropout(torch.tanh(h_left))
        h_right = torch.matmul(self.W_r, right_emb_T) + torch.matmul(torch.matmul(self.W_l, left_emb_T),
                                                                     affinity_matrix.t())
        h_right = self.dropout(torch.tanh(h_right))
        l_input = torch.matmul(self.w_hl, h_left)
        r_input = torch.matmul(self.w_hr, h_right)
        a_l = F.softmax(l_input, dim=1)  # 1 * N
        a_r = F.softmax(r_input, dim=1)  # 1 * T

        # global_left = torch.mm(a_l, left_emb).repeat(N, 1)   # N * d
        # global_right = torch.mm(a_r, right_emb).repeat(T, 1)
        # left_emb = self.alpha * left_emb + (1 - self.alpha) * global_left
        # right_emb = self.alpha * right_emb + (1 - self.alpha) * global_right
        global_left = torch.matmul(a_l, left_emb)  # 1 * d
        global_right = torch.matmul(a_r, right_emb)
        left_emb = self.alpha * left_emb + (1 - self.alpha) * torch.matmul(a_l.t().repeat(1, T), right_emb)
        right_emb = self.alpha * right_emb + (1 - self.alpha) * torch.matmul(a_r.t().repeat(1, N), left_emb)

        global_loss = torch.sum((global_left - global_right) ** 2, dim=1)
        # global_loss = None
        return global_loss, left_emb, right_emb


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag, use='node'):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        self.use = use
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            print("i={} layer: f_in={}, f_out={}".format(i, f_in, n_units[i + 1]))
            self.layer_stack.append(
                MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False, use=use))

    def forward(self, x, adj, weight=None):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            if self.use == 'node':
                x = gat_layer(x, adj)
            else:
                x = gat_layer(x, adj, weight)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, device=None):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        # self.highway = Highway(nout, nout, device)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = self.highway(x)
        return x


class Highway(nn.Module):
    def __init__(self, in_dim, out_dim, device=None):
        super(Highway, self).__init__()
        self.cuda = device
        self.fc1 = self.init_Linear(in_fea=in_dim, out_fea=out_dim, bias=True)
        self.gate_layer = self.init_Linear(in_fea=in_dim, out_fea=out_dim, bias=True)

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        if self.cuda is True:
            return linear.cuda()
        else:
            return linear

    def forward(self, x):
        in_fea = x.size(0)
        out_fea = x.size(1)
        normal_fc = torch.tanh(self.fc1(x))
        transformation_layer = torch.sigmoid(self.gate_layer(x))
        carry_layer = 1 - transformation_layer
        allow_transformation = torch.mul(normal_fc, transformation_layer)
        allow_carry = torch.mul(x, carry_layer)
        information_flow = torch.add(allow_transformation, allow_carry)
        return information_flow


class MultiViewEncoder(nn.Module):
    def __init__(self, args, device,
                 ent_num, rel_num, name_size,
                 rel_size, attr_size,
                 use_project_head=False):
        super(MultiViewEncoder, self).__init__()

        self.args = args
        self.device = device
        attr_dim = self.args.attr_dim
        rel_dim = self.args.rel_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.REL_NUM = rel_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.rel_n_units = [int(x) for x in self.args.rel_hidden_units.strip().split(",")]

        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.rel_n_heads = [int(x) for x in self.args.rel_heads.strip().split(",")]

        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])
        self.rel_input_dim = int(self.args.rel_hidden_units.strip().split(",")[0])
        self.ent_dim = 300
        self.rel_output_dim = self.rel_input_dim  # 100

        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout, device=self.args.cuda)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)
        if self.args.ent_name:
            print("with entity name")
            if self.args.word_embedding == "wc":
                self.wc_fc = nn.Linear(name_size, self.ent_dim, bias=False)
            if "SRPRS" in self.args.file_dir:
                self.tri_fc = nn.Linear(900, self.ent_dim, bias=False)
            else:
                self.tri_fc = nn.Linear(1100, self.ent_dim, bias=False)
        else:
            if "SRPRS" in self.args.file_dir:
                self.tri_fc = nn.Linear(600, self.ent_dim, bias=False)
            else:
                self.tri_fc = nn.Linear(800, self.ent_dim, bias=False)

        if self.args.w_triple_gat > 0:
            self.rel_emb = nn.Parameter(
                nn.init.sparse_(torch.empty(self.REL_NUM*2, self.rel_input_dim), sparsity=0.15)).to(device)

            self.highway1 = FinalHighway(self.ent_dim).to(device)
            self.highway2 = FinalHighway(self.ent_dim).to(device)
            # self.highway3 = FinalHighway(self.ent_dim).to(device)
            self.ea1 = EALayer(self.REL_NUM, 300, self.rel_output_dim, mode="add", use_ra=False).to(device)
            self.ea2 = EALayer(self.REL_NUM, 300, self.rel_output_dim, mode="add", use_ra=False).to(device)
            # self.ea3 = EALayer(self.REL_NUM, 300, self.rel_output_dim, mode="add", use_ra=self.args.w_ra).to(device)

        # Relation Embedding(for entity)
        self.rel_shared_fc = nn.Linear(self.REL_NUM, rel_dim)
        # self.emb_rel_fc = nn.Linear(200 * 2, 200)
        # Attribution Embedding(for entity)
        if self.args.w_attr:
            self.att_fc = nn.Linear(attr_size, attr_dim)

        if self.use_project_head:
            # self.img_pro = ProjectionHead(img_dim, img_dim, img_dim, dropout)
            self.att_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.rel_pro = ProjectionHead(400, 400, 400, dropout)
            self.gph_pro = ProjectionHead(self.n_units[2], self.n_units[2], self.n_units[2], dropout)

        self.fusion = MultiViewFusion(modal_num=self.args.inner_view_num,
                                      with_weight=self.args.with_weight)

    def forward(self,
                input_idx, r_input_idx,
                e_in, e_out, e_adj, epoch,
                r_in_adj, r_out_adj,r_path_adj,
                edge_index_all, rel_all, name_emb,
                rel_features_in=None,
                rel_features_out=None,
                att_features=None):

        if self.args.w_gph:
            gph_emb = self.cross_graph_model(self.entity_emb(input_idx), e_adj)
            # gph_emb = self.cross_graph_model(name_emb, e_adj)
        else:
            gph_emb = None

        if self.args.w_triple_gat:
            rel_emb = self.rel_emb

        if self.args.w_rel:
            rel_in_f = self.rel_shared_fc(rel_features_in)
            rel_out_f = self.rel_shared_fc(rel_features_out)
            ent_rel_features = torch.cat([rel_in_f, rel_out_f], dim=1)
            ent_rel_emb = ent_rel_features
            # ent_rel_emb = self.emb_rel_fc(ent_rel_features)
        else:
            ent_rel_emb = None

        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None
        if self.args.ent_name > 0 and self.args.word_embedding == "wc":
            name_emb = self.wc_fc(name_emb)
        joint_emb = self.fusion([name_emb, gph_emb, ent_rel_emb, att_emb])
        
        if self.args.w_triple_gat:
            joint_emb = self.tri_fc(joint_emb)
            res_att = None
            x_e, rel_emb, res_att = self.ea1(joint_emb, edge_index_all, rel_all, rel_emb, res_att)
            x_e_1 = self.highway1(joint_emb, x_e)
            # x_e_1 = x_e

            x_e, rel_emb, res_att = self.ea2(x_e_1, edge_index_all, rel_all, rel_emb, res_att)
            x_e_2 = self.highway2(x_e_1, x_e)
            # x_e_2 = x_e
            # x_e, rel_emb, res_att = self.ea3(x_e_2, edge_index_all, rel_all, rel_emb, res_att)
            # x_e_3 = self.highway3(x_e_2, x_e)
            tri_emb = torch.cat([x_e_1, x_e_2], dim=1) 
            # tri_emb = x_e_1
        else:
            rel_emb = None
            tri_emb = None

        return gph_emb, ent_rel_emb, rel_emb, att_emb, joint_emb, tri_emb, name_emb


def pooling(ent_rel_list, method="avg"):
    # len = ent_rel_list.shape[0]
    if method == "avg":
        return torch.mean(ent_rel_list, dim=0).unsqueeze(0)
    elif method == 'max':
        return torch.max(ent_rel_list, 0)[0].unsqueeze(0)
    elif method == 'min':
        return torch.min(ent_rel_list, 0)[0].unsqueeze(0)


class MultiViewFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)),
                                   requires_grad=self.requires_grad)

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
        joint_emb = torch.cat(embs, dim=1)
        return joint_emb


class FinalHighway(nn.Module):
    def __init__(self, x_hidden):
        super(FinalHighway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        x = torch.mul(gate, x2) + torch.mul(1 - gate, x1)
        return x


class EALayer(nn.Module):
    def __init__(self, rel_num, e_hidden, r_hidden, mode="add", use_ra=False):
        super(EALayer, self).__init__()
        print("using EALayer, mode={}".format(mode))
        self.use_ra = use_ra
        
        self.mode = mode
        self.ww = nn.Linear(r_hidden, e_hidden, bias=False)
        self.rel = nn.Linear(r_hidden, r_hidden, bias=False)
        # self.tail = nn.Linear(e_hidden, e_hidden, bias=False)
        if self.use_ra:
            self.ra_layer = RALayer(e_hidden=e_hidden, r_hidden=r_hidden)
        if self.mode == "cross":
            self.rel_weight = nn.Parameter(nn.init.xavier_normal_(torch.empty(2*rel_num, e_hidden)))
        elif self.mode == "concat":
            self.cat_fc = nn.Linear(e_hidden, e_hidden*2, bias=False)
            self.out_fc = nn.Linear(e_hidden*2, e_hidden, bias=False)
        else:
            pass
        # self.e_in = nn.Linear(e_hidden+r_hidden, r_hidden, bias=False)
        # self.e_out = nn.Linear(e_hidden+r_hidden, e_hidden, bias=False)

    def forward(self, x, edge_index, edge_type, rel_emb, res_att):
        if self.use_ra:
            rel_emb, res_att = self.ra_layer(x, edge_index, edge_type, rel_emb, res_att)
        r_emb = self.ww(rel_emb)
        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]
        e_head = x[edge_index_i]
        e_rel = r_emb[edge_type]
        if self.mode == "add":
            h_r = e_head + e_rel
        elif self.mode == "sub":
            h_r = e_head - e_rel
        elif self.mode == "cross":
            rel_weight = torch.index_select(self.rel_weight, 0, edge_type)
            h_r = e_head * e_rel * rel_weight + e_head * rel_weight
        elif self.mode == "concat":
            x = self.cat_fc(x)
            h_r = torch.cat([e_head, e_rel], dim=1)
        else:
            pass
        e_tail = x[edge_index_j]
        dp_att = torch.sum(h_r * e_tail, dim=-1)
        attention_weights = torch.softmax(dp_att, dim=-1)
        x_e = scatter(h_r * torch.unsqueeze(attention_weights, dim=-1), edge_index_j, dim=0, reduce='sum')
        if self.mode == "concat":
            x_e = self.out_fc(x_e)
        x_e = F.relu(x_e)
        return x_e, self.rel(rel_emb), res_att
    