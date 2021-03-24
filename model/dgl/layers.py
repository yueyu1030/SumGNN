"""
File baseed off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


class RGCNLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, aggregator, bias=None, activation=None, num_nodes=122343142,dropout=0.0, 
        edge_dropout=0.0, is_input_layer=False, embed=False, add_transe_emb=True):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))
        self.add_transe_emb = add_transe_emb
        self.aggregator = aggregator
        self.is_input_layer = is_input_layer
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = Identity()
        #if is_input_layer:
        if embed is not None:
            self.embed = embed
        elif self.is_input_layer and self.add_transe_emb:
            self.embed = nn.Parameter(torch.Tensor(self.num_nodes, self.out_dim), requires_grad = True)
            nn.init.xavier_uniform_(self.embed,
                                    gain=nn.init.calculate_gain('relu'))


    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, attn_rel_emb=None):

        self.propagate(g, attn_rel_emb)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr
        if self.is_input_layer and self.add_transe_emb:
            x = torch.cat([self.embed[g.ndata['idx']], g.ndata['h']], dim = 1)

            g.ndata['repr'] = x.unsqueeze(1).reshape(-1, 2, self.out_dim)
            #print(x.shape, g.ndata['repr'].shape)
        elif self.is_input_layer :
            g.ndata['repr'] = g.ndata['h'].unsqueeze(1)
        else:
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)
            # print(g.ndata['repr'], g.ndata['repr'].shape)
            # assert 0


class RGCNBasisLayer(RGCNLayer):
    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels,num_bases=-1,  num_nodes = 12345342, bias=None, has_kg = True,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False, embed = None, transformer = True, add_transe_emb=True,
                 one_attn = False, A1 = None, A2 = None, gamma= 0.0):
        super(
            RGCNBasisLayer,
            self).__init__(
            inp_dim,
            out_dim,
            aggregator,
            bias,
            activation,
            num_nodes = num_nodes,
            dropout=dropout,
            edge_dropout=edge_dropout,
            is_input_layer=is_input_layer,
            embed = embed,
            add_transe_emb = add_transe_emb)
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn
        self.has_kg = has_kg
        self.num_nodes = num_nodes
        self.transformer = transformer
        self.add_transe_emb = add_transe_emb
        self.gamma = gamma
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        # self.weight = basis_weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        self.one_attn = one_attn

        #print(self.weight.shape)
        if self.has_attn:
            if self.transformer:
                if self.one_attn == False:
                #else:
                    self.A1 = nn.Linear(self.inp_dim, self.attn_rel_emb_dim)
                    self.A2 = nn.Linear(self.inp_dim, self.attn_rel_emb_dim)
                else:
                    self.A1 = A1
                    self.A2 = A2
            else:
                self.A = nn.Linear(2 * self.inp_dim + 1 * self.attn_rel_emb_dim, inp_dim)
                self.B = nn.Linear(inp_dim, 1)

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g, attn_rel_emb = None, nonKG = True):
        # generate all weights from bases
        #print(g.ndata['idx'], g.ndata['idx'].shape, g.ndata['feat'].shape)

        weight = self.weight.view(self.num_bases,
                                  self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(
            self.num_rels, self.inp_dim, self.out_dim)

        g.edata['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))

        input_ = 'feat' if self.is_input_layer else 'h'

        def msg_func(edges):
            w = weight.index_select(0, edges.data['type'])
            if input_ == 'feat' and self.add_transe_emb:
                x = torch.cat( [
                                edges.src[input_], 
                                self.embed[edges.src['idx']]
                                ], dim = 1
                            )
                # else:
                #     x =  edges.src[input_]
                # msg = edges.data['w'] * torch.bmm(x.unsqueeze(1), w).squeeze(1)
            else:
                x = edges.src[input_]
            msg = edges.data['w'] * torch.bmm(x.unsqueeze(1), w).squeeze(1)
            curr_emb = torch.mm(x, self.self_loop_weight)  # (B, F)
            # if self.inp_dim == self.out_dim:
            #     curr_emb += x
            #print(curr_emb.shape)
            if self.has_attn:
                if input_ == 'feat' and self.add_transe_emb:
                    y = torch.cat( [edges.dst[input_], self.embed[edges.dst['idx']]], dim = 1)
                    # else:
                    #     y = edges.dst[input_]
                else:
                    #x = edges.src[input_]
                    y = edges.dst[input_]
                if self.transformer:
                    import math
                    #x_feat = self.A1.index_select(0, edges.src['ntype']) # B * b_in * b_out
                    #y_feat = self.A1.index_select(0, edges.dst['ntype'])
                    #x_hat = torch.bmm(x.unsqueeze(1), x_feat).squeeze(1)
                    #y_hat = torch.bmm(y.unsqueeze(1), y_feat).squeeze(1)+attn_rel_emb(edges.data['type']) # B * b_out
                    if self.one_attn:
                        x = torch.cat( [edges.src['feat'], self.embed[edges.src['idx']]], dim = 1)
                        y = torch.cat( [edges.dst['feat'], self.embed[edges.dst['idx']]], dim = 1)
                        x_hat = self.A1(x)
                        y_hat = self.A2(y)+attn_rel_emb(edges.data['type'])
                        a = F.relu(torch.tanh(torch.sum(x_hat * y_hat/math.sqrt(x.shape[1]), dim = 1))).unsqueeze(1)         
                    else:
                        x_hat = self.A1(x)
                        y_hat = self.A2(y)+attn_rel_emb(edges.data['type'])
                        a = torch.tanh(F.relu(torch.sum(x_hat * y_hat/math.sqrt(self.inp_dim), dim = 1))).unsqueeze(1)
                    #msg += attn_rel_emb(edges.data['type'])
                    if 1:
                        b = a.detach().cpu().numpy()
                        c = edges.src['idx'].detach().cpu().numpy().reshape(-1,1)
                        d = edges.dst['idx'].detach().cpu().numpy().reshape(-1,1)
                        e = edges.src['id'].detach().cpu().numpy().reshape(-1,1)
                        f = edges.dst['id'].detach().cpu().numpy().reshape(-1,1)
                        g = edges.data['type'].detach().cpu().numpy().reshape(-1,1)
                       # print(b.shape, c.shape, d.shape, e.shape, f.shape)
                        z= np.concatenate([b,c,d,e,f, g], axis = 1)
                        np.savetxt('Drugbank/%.3f.txt'%(np.random.rand()*100), z, '%.4f')

                    a = torch.where(a > self.gamma, a, torch.zeros(a.shape).to(a.device)) 
                    #print(a)
                else:
                    e = torch.cat([x, y , attn_rel_emb(edges.data['type'])], dim=1)
                    a = torch.sigmoid(self.B(F.relu(self.A(e))))

            else:
                a = torch.ones((len(edges), 1)).to(device=w.device)

            if not self.has_kg:
                a = edges.src["mask"].reshape(-1,1)*edges.dst["mask"].reshape(-1,1)*a
            #print(curr_emb)
            #print( g, msg.shape, a.shape)
            #assert 0
            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

        g.update_all(msg_func, self.aggregator, None)
