import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import json
from torch.nn.utils import clip_grad_norm_


class Trainer():
    def __init__(self, params, graph_classifier, train, train_evaluator = None, valid_evaluator=None, test_evaluator = None):
        self.graph_classifier = graph_classifier
        self.train_evaluator=train_evaluator
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train
        self.test_evaluator = test_evaluator
        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        # self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        if params.dataset == 'ddi':
            self.criterion = nn.CrossEntropyLoss()
        elif params.dataset == 'ddi2':
            self.criterion = nn.BCELoss(reduce=False) 
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def load_model(self):
        self.graph_classifier.load_state_dict(torch.load("my_resnet.pth"))

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        bar = tqdm(enumerate(dataloader))
        for b_idx, batch in bar:
            #data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
            
            #try:
            self.optimizer.zero_grad()
            score_pos = self.graph_classifier(data_pos)
            #score_neg = self.graph_classifier(data_neg)
            #loss = self.criterion(score_pos, score_pos, torch.Tensor([1]).to(device=self.params.device))
            if self.params.dataset == 'ddi':
                loss = self.criterion(score_pos, r_labels_pos)
            elif self.params.dataset == 'ddi2':
                m = nn.Sigmoid()
                score_pos = m(score_pos)
                targets_pos = targets_pos.unsqueeze(1)
                #print(score_pos.shape, r_labels_pos.shape)
                loss_train = self.criterion(score_pos, r_labels_pos * targets_pos)
                loss = torch.sum(loss_train * r_labels_pos)
            # print(score_pos, score_neg, loss)
            
            loss.backward()
            clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.updates_counter += 1
            bar.set_description('epoch: ' + str(b_idx+1) + '/ loss_train: ' + str(loss.cpu().detach().numpy()))
    
            # except RuntimeError:
            #     print(data_pos, r_labels_pos, targets_pos)
            #    print('-------runtime error--------')
            #    continue
            with torch.no_grad():
                # all_scores += score_pos.squeeze().detach().cpu().tolist() #+ score_neg.squeeze().detach().cpu().tolist()
                # all_labels += targets_pos.tolist() #+ targets_neg.tolist()
                total_loss += loss.item()
                if self.params.dataset != 'ddi2':
                    
                    label_ids = r_labels_pos.to('cpu').numpy()
                    all_labels += label_ids.flatten().tolist()
                    #y_pred = y_pred + F.softmax(output, dim = -1)[:, -1].cpu().flatten().tolist()
                    #outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
                    all_scores += torch.argmax(score_pos, dim=1).cpu().flatten().tolist() 
            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                result, save_dev_data = self.valid_evaluator.eval()
                test_result, save_test_data = self.test_evaluator.eval()
                logging.info('\033[95m Eval Performance:' + str(result) + 'in ' + str(time.time() - tic)+'\033[0m')
                logging.info('\033[93m Test Performance:' + str(test_result) + 'in ' + str(time.time() - tic)+'\033[0m')
                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0
                    if self.params.dataset != 'ddi2':
                        logging.info('\033[93m Test Performance Per Class:' + str(save_test_data) + 'in ' + str(time.time() - tic)+'\033[0m')
                    else:
                        with open('experiments/%s/result.json'%(self.params.experiment_name), 'a') as f:
                            f.write(json.dumps(save_test_data))
                            f.write('\n')
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc']
        weight_norm = sum(map(lambda x: torch.norm(x), model_params))
        if self.params.dataset != 'ddi2':
            auc = metrics.f1_score(all_labels, all_scores, average='macro')
            auc_pr = metrics.f1_score(all_labels, all_scores, average='micro')

            return total_loss/b_idx, auc, auc_pr, weight_norm
        else:
            return total_loss/b_idx, 0, 0, weight_norm

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            
            loss, auc, auc_pr, weight_norm = self.train_epoch()

            # loss = 0
            # auc = 0
            # auc_pr = 0
            # weight_norm = 0
            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            # if self.valid_evaluator and epoch % self.params.eval_every == 0:
            #     result = self.valid_evaluator.eval()
            #     logging.info('\nPerformance:' + str(result))
            
            #     if result['auc'] >= self.best_metric:
            #         self.save_classifier()
            #         self.best_metric = result['auc']
            #         self.not_improved_count = 0

            #     else:
            #         self.not_improved_count += 1
            #         if self.not_improved_count > self.params.early_stop:
            #             logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
            #             break
            #     self.last_metric = result['auc']

            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

    def case_study(self):
        self.reset_training_state()
        test_result, save_test_data = self.test_evaluator.print_result()
        # test_result, save_test_data = self.train_evaluator.print_attn_weight()
        # test_result, save_test_data = self.test_evaluator.print_attn_weight()
    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')
