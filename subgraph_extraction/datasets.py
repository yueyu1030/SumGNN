from torch.utils.data import Dataset
import timeit
import os
import logging
import lmdb
import numpy as np
import json
import pickle
import dgl
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, process_files_ddi, save_to_file, plot_rel_dist,process_files_decagon
from .graph_sampler import *
import pdb


def generate_subgraph_datasets(params, splits=['train', 'valid', 'test'], saved_relation2id=None, max_label_value=None):

    testing = 'test' in splits
    #adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = process_files(params.file_paths, saved_relation2id)
    
    triple_file = 'data/{}/relations_2hop.txt'.format(params.dataset)
    if params.dataset == 'drugbank':
        adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = process_files_ddi(params.file_paths, triple_file, saved_relation2id)
    else:
        adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr = process_files_decagon(params.file_paths, triple_file, saved_relation2id)
    # plot_rel_dist(adj_list, os.path.join(params.main_dir, f'data/{params.dataset}/rel_dist.png'))
    #print(triplets.keys(), triplets_mr.keys())
    data_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
    if not os.path.isdir(data_path) and testing:
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:
        if params.dataset == 'drugbank':
            graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}
        elif params.dataset == 'BioSNAP':
            graphs[split_name] = {'triplets': triplets_mr[split_name], 'max_size': params.max_links, "polarity_mr": polarity_mr[split_name]}
    # Sample train and valid/test links
    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link, max_size=split['max_size'], constrained_neg_prob=params.constrained_neg_prob)
    #print(graphs.keys())
    if testing:
        directory = os.path.join(params.main_dir, 'data/{}/'.format(params.dataset))
        save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'], id2entity, id2relation)

    links2subgraphs(adj_list, graphs, params, max_label_value)


def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None, add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='', kge_model='', file_name='', \
        ssp_graph = None,  relation2id= None, id2entity= None, id2relation= None, rel= None,  graph = None, morgan_feat = None):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name
        triple_file = 'data/{}/relations_2hop.txt'.format(dataset)
        self.entity_type = np.loadtxt('data/{}/entity.txt'.format(dataset))

        if not ssp_graph:
            if dataset == 'drugbank':
                ssp_graph, triplets, entity2id, relation2id, id2entity, id2relation, rel = process_files_ddi(raw_data_paths, triple_file, included_relations)
            else:
                ssp_graph, triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr = process_files_decagon(raw_data_paths, triple_file, included_relations)

            
            data_path =  'data/{}/relation2id.json'.format(dataset)
            #print(os.pwd)
            if not os.path.isdir(data_path):
                with open(data_path, 'w') as f:
                    json.dump(relation2id, f)
            self.num_rels = rel
            print('number of relations:%d'%(self.num_rels))

            # Add transpose matrices to handle both directions of relations.
            if add_traspose_rels:
                ssp_graph_t = [adj.T for adj in ssp_graph]
                ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
            self.aug_num_rels = len(ssp_graph)
            self.graph = ssp_multigraph_to_dgl(ssp_graph)
            self.ssp_graph = ssp_graph
        else:
            self.ssp_graph = ssp_graph
            self.graph = graph
            self.aug_num_rels = len(ssp_graph)
            self.num_rels = rel

        self.id2entity = id2entity
        self.id2relation = id2relation


        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        lst = []
        def json_save(data, dataset_name):
            with open(dataset_name, 'w') as f:
                f.write(json.dumps(data, indent = 4))
        json_save(lst, db_name_pos+'3.json')
        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            #print(nodes_pos, r_label_pos, g_label_pos, n_labels_pos)
            #print(nodes_pos, r_label_pos, g_label_pos, n_labels_pos)
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)

        return subgraph_pos, g_label_pos, r_label_pos

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):

        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))
        #print(subgraph, subgraph.nodes(), subgraph.ndata, subgraph.edges(), subgraph.edata)
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
                
        subgraph.ndata['idx'] = torch.LongTensor(np.array(nodes))
        subgraph.ndata['ntype'] = torch.LongTensor(self.entity_type[nodes])
        subgraph.ndata['mask'] = torch.LongTensor(np.where(self.entity_type[nodes]==1, 1, 0))
        try:
            edges_btw_roots = subgraph.edge_id(0, 1)
            rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        except AssertionError:
            pass

        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats)
        try:
            edges_btw_roots = subgraph.edge_id(0, 1)
            subgraph.remove_edges(edges_btw_roots)
        except AssertionError:
            pass
        return subgraph #, torch.LongTensor([head_idx, tail_idx])

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids) 

        self.n_feat_dim = n_feats.shape[1]  
        return subgraph#, h__, t__

