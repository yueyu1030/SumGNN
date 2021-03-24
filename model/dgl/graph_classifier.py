from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import torch.nn.functional as F
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.dropout = nn.Dropout(p = 0.3)
        self.relu = nn.ReLU()
        #print(len(relation2id))
        self.train_rels = params.train_rels
        self.relations = params.num_rels
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        #self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.mp_layer1 = nn.Linear(self.params.feat_dim, self.params.emb_dim)
        self.mp_layer2 = nn.Linear(self.params.emb_dim, self.params.emb_dim)
        if self.params.add_ht_emb and self.params.add_sb_emb:
            if self.params.add_feat_emb and self.params.add_transe_emb:
                self.fc_layer = nn.Linear(3 * (1+self.params.num_gcn_layers) * self.params.emb_dim + 2*self.params.emb_dim, self.train_rels)
            elif self.params.add_feat_emb :
                self.fc_layer = nn.Linear(3 * (self.params.num_gcn_layers) * self.params.emb_dim + 2*self.params.emb_dim, self.train_rels)
            else:
                self.fc_layer = nn.Linear(3 * (1+self.params.num_gcn_layers) * self.params.emb_dim, self.train_rels)
        elif self.params.add_ht_emb:
            self.fc_layer = nn.Linear(2 * (1+self.params.num_gcn_layers) * self.params.emb_dim, self.train_rels)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim, self.train_rels)
        #print(self.fc_layer)
    def drug_feat(self, emb):
        self.drugfeat = emb

    def forward(self, data):
        g = data
        g.ndata['h'] = self.gnn(g)
        #print('repr:',g.ndata['repr'], g.ndata['repr'].shape)
        #assert 0
        g_out = mean_nodes(g, 'repr')
        #print('g_out', g_out.shape)
        #assert 0
        # print(g_out.shape,g.ndata['h'].shape)
        
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]
        #print(g.ndata['idx'][head_ids], g.ndata['idx'][tail_ids],  g.ndata['idx'][tail_ids].shape)
        head_feat = self.drugfeat[g.ndata['idx'][head_ids]]
        tail_feat = self.drugfeat[g.ndata['idx'][tail_ids]]
        #print(head_feat.shape, tail_feat.shape)
        # drug_feat = self.drugfeat[drug_idx]
        # print(drug_feat, drug_feat.shape)
        if self.params.add_feat_emb:
            fuse_feat1 = self.mp_layer2( self.relu( self.dropout( self.mp_layer1(
                            head_feat #torch.cat([head_feat, tail_feat], dim = 1)
                        ))))
            fuse_feat2 = self.mp_layer2( self.relu( self.dropout( self.mp_layer1(
                            tail_feat #torch.cat([head_feat, tail_feat], dim = 1)
                        ))))
            fuse_feat = torch.cat([fuse_feat1, fuse_feat2], dim = 1)
        if self.params.add_ht_emb and self.params.add_sb_emb:
            if self.params.add_feat_emb and self.params.add_transe_emb:
                g_rep = torch.cat([g_out.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   head_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   tail_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   fuse_feat.view(-1, 2*self.params.emb_dim)
                                   ], dim=1)
            elif self.params.add_feat_emb:
                g_rep = torch.cat([g_out.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                                   head_embs.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                                   tail_embs.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                                   fuse_feat.view(-1, 2*self.params.emb_dim)
                                   ], dim=1)
            else:
                g_rep = torch.cat([g_out.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   head_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   tail_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                   #fuse_feat.view(-1, 2*self.params.emb_dim)
                                   ], dim=1)
            
        elif self.params.add_ht_emb:
            g_rep = torch.cat([
                                head_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                tail_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim)
                               ], dim=1)
        else:
            g_rep = g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim)
        #print(g_rep.shape, self.params.add_ht_emb, self.params.add_sb_emb)
        output = self.fc_layer(F.dropout(g_rep, p =0.3))
        # print(head_ids.detach().cpu().numpy(), tail_ids.detach().cpu().numpy())
        return output
