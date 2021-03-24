import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl, move_batch_to_device_dgl_ddi2

from model.dgl.graph_classifier import GraphClassifier as dgl_model

from managers.evaluator import Evaluator, Evaluator_ddi2
from managers.trainer import Trainer
import numpy as np
from warnings import simplefilter
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file)
    print(train.graph)
    #assert 0
    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file,
                            ssp_graph = train.ssp_graph, 
                            id2entity= train.id2entity, id2relation= train.id2relation, rel= train.num_rels,  graph = train.graph)
    test = SubgraphDataset(params.db_path, 'test_pos', 'test_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file,
                            ssp_graph = train.ssp_graph,  
                            id2entity= train.id2entity, id2relation= train.id2relation, rel= train.num_rels,  graph = train.graph)
    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.inp_dim = train.n_feat_dim
    params.train_rels = 200 if params.dataset == 'ddi2' else params.num_rels
    params.num_nodes = 35000

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label
    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    graph_classifier = initialize_model(params, dgl_model, params.load_model)
    if params.dataset == 'ddi':
        if params.feat == 'morgan':
            import pickle 
            with open('data/{}/DB_molecular_feats.pkl'.format(params.dataset), 'rb') as f:
                x = pickle.load(f, encoding='utf-8')
            mfeat =  []
            for y in x['Morgan_Features']:
                mfeat.append(y)
            params.feat_dim = 1024
        elif  params.feat == 'pca':
            mfeat = np.loadtxt('data/{}/PCA.txt'.format(params.dataset))
            params.feat_dim = 200
        elif  params.feat == 'pretrained':
            mfeat = np.loadtxt('data/{}/pretrained.txt'.format(params.dataset))
            params.feat_dim = 200
    elif params.dataset == 'ddi2':
        mfeat = []
        rfeat = []
        import pickle 
        with open('data/{}/id2drug_feat.pkl'.format(params.dataset), 'rb') as f:
            x = pickle.load(f, encoding='utf-8') 
        for z in x:
            y = x[z]['Morgan']
            mfeat.append(y)
            y = x[z]['rdkit2d']
            rfeat.append(y)
            params.feat_dim = 1024

    graph_classifier.drug_feat(torch.FloatTensor(np.array(mfeat)).to(params.device))
    
    valid_evaluator = Evaluator(params, graph_classifier, valid) if params.dataset == 'ddi' else Evaluator_ddi2(params, graph_classifier, valid)
    test_evaluator = Evaluator(params, graph_classifier, test) if params.dataset == 'ddi' else Evaluator_ddi2(params, graph_classifier, test)
    train_evaluator = Evaluator(params, graph_classifier, train) if params.dataset == 'ddi' else Evaluator_ddi2(params, graph_classifier, valid)
    
    trainer = Trainer(params, graph_classifier, train, train_evaluator, valid_evaluator,test_evaluator)

    logging.info('Starting training with full batch...')
    trainer.case_study()
    #trainer.train()



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default1",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--gpu", type=int, default=2,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="dev",
                        help="Name of file containing validation triplets")
    parser.add_argument("--test_file", "-ttf", type=str, default="test",
                        help="Name of file containing validation triplets")
    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=50,
                        help="Learning rate of the optimizer")
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Interval of epochs to evaluate the model?")
    parser.add_argument("--eval_every_iter", type=int, default=526,
                        help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=100,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=1e-5,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=250000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=200,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=0,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')

    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32,
                        help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=2,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.4,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--add_sb_emb', '-sb', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True,
                        help='whether to have attn in model or not')
    parser.add_argument('--has_kg', '-kg', type=bool, default=True,
                        help='whether to have kg in model or not')
    parser.add_argument('--feat', '-f', type=str, default='morgan',
                        help='whether to have attn in model or not')
    parser.add_argument('--feat_dim', type=int, default=1024,
                        help='whether to have attn in model or not')
    parser.add_argument('--add_feat_emb', '-feat', type=bool, default=True,
                        help='whether to have attn in model or not')
    parser.add_argument('--add_transe_emb', type=bool, default=True,
                        help='whether to have attn in model or not')
    parser.add_argument('--one_attn', type=bool, default=False,
                        help='whether to have attn in model or not')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='whether to have attn in model or not')
    params = parser.parse_args()
    initialize_experiment(params, __file__)

    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_file)),
        'test': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.test_file))
    }

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl if params.dataset == 'ddi' else move_batch_to_device_dgl_ddi2

    main(params)
