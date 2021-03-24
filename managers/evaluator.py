import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import  cohen_kappa_score, accuracy_score
from tqdm import tqdm
class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def print_attn_weight(self):
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                s = r_labels_pos.cpu().numpy().tolist()
                # print(s)
                #if s[0] in [ 0,  6, 12, 16, 17, 18, 21, 22, 30, 34, 35, 37, 40, 41, 43, 44, 45, 47, 49, 50, 51, 54, 55, 58, 61, 64, 65, 77, 80, 83, 85] or s[1] in [ 0,  6, 12, 16, 17, 18, 21, 22, 30, 34, 35, 37, 40, 41, 43, 44,45, 47, 49, 50, 51, 54, 55, 58, 61, 64, 65, 77, 80, 83, 85]:
                if 19 in s:
                    print(s, targets_pos)
                    score_pos = self.graph_classifier(data_pos)
                    s = score_pos.detach().cpu().numpy()
                    # with open('Drugbank/result.txt', 'a') as f:
                    #     f.write()

    def print_result(self):
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.eval()
        pos_labels = []
        pos_argscores = []
        pos_scores = []
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                score_pos = self.graph_classifier(data_pos)
                label_ids = r_labels_pos.to('cpu').numpy()
                pos_labels += label_ids.flatten().tolist()
                pos_argscores += torch.argmax(score_pos, dim=1).cpu().flatten().tolist() 
                print( torch.max(score_pos, dim=1, out=None))
                pos_scores += torch.max(score_pos, dim=1)[0].cpu().flatten().tolist() 
                # s = r_labels_pos.cpu().numpy().tolist()
                # # print(s)
                # #if s[0] in [ 0,  6, 12, 16, 17, 18, 21, 22, 30, 34, 35, 37, 40, 41, 43, 44, 45, 47, 49, 50, 51, 54, 55, 58, 61, 64, 65, 77, 80, 83, 85] or s[1] in [ 0,  6, 12, 16, 17, 18, 21, 22, 30, 34, 35, 37, 40, 41, 43, 44,45, 47, 49, 50, 51, 54, 55, 58, 61, 64, 65, 77, 80, 83, 85]:
                # if 19 in s:
                #     print(s, targets_pos)
                #     score_pos = self.graph_classifier(data_pos)
                #     s = score_pos.detach().cpu().numpy()
        with open('Drugbank/results.txt', 'w') as f:
            for (x,y,z) in zip(pos_argscores, pos_labels, pos_scores):
                f.write('%d %d %d\n'%(x, y, z))


    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        y_pred = []
        label_matrix = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):

                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                score_pos = self.graph_classifier(data_pos)
                #score_neg = self.graph_classifier(data_neg)

                # preds += torch.argmax(logits.detach().cpu(), dim=1).tolist()
                label_ids = r_labels_pos.to('cpu').numpy()
                pos_labels += label_ids.flatten().tolist()
                #y_pred = y_pred + F.softmax(output, dim = -1)[:, -1].cpu().flatten().tolist()
                #outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
                pos_scores += torch.argmax(score_pos, dim=1).cpu().flatten().tolist() 

                # pred = F.softmax(score_pos, dim = -1).detach().cpu().numpy()
                # label_mat = np.zeros(pred.shape)
                # label_mat[np.arange(label_mat.shape[0]), label_ids] = 1
                # y_pred.append(pred)
                # label_matrix.append(label_mat)

        # acc = metrics.accuracy_score(labels, preds)
        auc = metrics.f1_score(pos_labels, pos_scores, average='macro')
        auc_pr = metrics.f1_score(pos_labels, pos_scores, average='micro')
        f1 = metrics.f1_score(pos_labels, pos_scores, average=None)
        kappa = metrics.cohen_kappa_score(pos_labels, pos_scores)

        # y_pred = np.vstack(y_pred)
        # label_matrix = np.vstack(label_matrix)
        # #print(y_pred.T[0])
        # auprc = [average_precision_score(y_l, y_p) for (y_l, y_p) in zip(label_matrix.T ,  y_pred.T) if np.sum(y_l)>=2]
        # auroc = [roc_auc_score(y_l, y_p) for (y_l, y_p) in zip(label_matrix.T ,  y_pred.T) if np.sum(y_l)>=2]
        
        #print(s)
        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset, self.data.file_name, self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'auc': auc, 'microf1': auc_pr, 'k':kappa}, {'f1': f1}

class Evaluator_ddi2():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []

        y_pred = []
        y_label = []
        outputs = []

        pred_class = {}

        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):

                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                score_pos = self.graph_classifier(data_pos)

                m = nn.Sigmoid()
                #loss_fct = nn.BCELoss()
                pred = m(score_pos)
                #loss = loss_fct(pred, label)
                labels = r_labels_pos.detach().to('cpu').numpy() # batch * 200
                preds = pred.detach().to('cpu').numpy() # batch * 200
                targets_pos = targets_pos.detach().to('cpu').numpy()
                for (label_ids, pred, label_t) in zip(labels, preds, targets_pos):
                # label_ids = [x for x in label.detach().to('cpu').numpy() if x==1] # batch * 200
                # preds = pred.detach().to('cpu').numpy()[label_ids] # batch * 200
                    #print(label_ids, pred, label_t)
                    for i, (l, p) in enumerate(zip(label_ids, pred)):
                        #print(i, l, p)
                        if l == 1:
                            if i in pred_class:
                                pred_class[i]['pred'] += [p]
                                pred_class[i]['l'] += [label_t] 
                                pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
                            else:
                                pred_class[i] = {'pred':[p], 'l':[label_t], 'pred_label':[1 if p > 0.5 else 0]}


                # output = np.where(preds>0.5, 1, 0) # batch * 200
                # outputs.append(output)
                # y_label.append(label_ids)
                # y_pred.append(preds)

                #preds += torch.argmax(logits.detach().cpu(), dim=1).tolist()
                # label_ids = r_labels_pos.to('cpu').numpy()
                # pos_labels += label_ids.flatten().tolist()

                # pos_scores += torch.argmax(score_pos, dim=1).cpu().flatten().tolist() 

                #y_pred = y_pred + F.softmax(output, dim = -1)[:, -1].cpu().flatten().tolist()
                #outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
        # acc = metrics.accuracy_score(labels, preds)
        # auc = metrics.f1_score(pos_labels, pos_scores, average='macro')
        # auc_pr = metrics.f1_score(pos_labels, pos_scores, average='micro')
        roc_auc = [ roc_auc_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
        prc_auc = [ average_precision_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
        ap =  [accuracy_score(pred_class[l]['l'], pred_class[l]['pred_label']) for l in pred_class]



        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset, self.data.file_name, self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'auc': np.mean(roc_auc), 'auc_pr': np.mean(prc_auc), 'f1': np.mean(ap)}, {"auc_all":roc_auc,"aupr_all":prc_auc, "f1_all":ap}

