import time
import numpy as np
# from document_reader import *
import os
import os.path
from os import path
from os import listdir
from os.path import isfile, join
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, get_constant_schedule
from metric import metric, CM_metric
import json
from json import JSONEncoder
#import notify
#from notify_message import *
#from notify_smtp import *
from util import *
from tqdm import tqdm
import wandb

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
class exp:
    def __init__(self, cuda, model, epochs, learning_rate, train_dataloader, valid_dataloader, test_dataloader, 
                 dataset, best_PATH, load_model_path, dpn, warmup_ratio=0.1, scheduler_type="constant",
                 model_name = None, relation_stats = None, lambdas = None, accum_iter=1, f1_metric='micro_f1'):
        self.cuda = cuda
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dpn = dpn
        self.accum_iter = accum_iter
        self.warmup_ratio = warmup_ratio
        self.scheduler_type = scheduler_type

        label_weights = []
        if relation_stats:
            for label in relation_stats.keys():
                label_weights.append(relation_stats[label])
            self.relation_stats = [w / sum(label_weights) for w in label_weights]
        if lambdas:
            self.lambda_1 = lambdas[0]
            self.lambda_2 = lambdas[1]
        if self.dpn == 1:
            self.out_class = 3
        else:
            self.out_class = 4
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        
        self.metric = f1_metric
        self.best_f1 = -0.000001
        self.best_macro_F1 = -0.000001
        self.best_cm = []
        self.best_PATH = best_PATH # to save model params here
        
        self.best_epoch = 0
        self.load_model_path = load_model_path # load pretrained model parameters for testing, prediction, etc.
        self.model_name = model_name
        self.file = open("./rst_file/" + model_name + ".rst", "a")

    def train(self, additional_eval_loader_dict={}):
        """
            additional_eval_loader_dict: the dict of evaluation data loaders. 
        """
        total_t0 = time.time()
        log_eval_best = {}

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True) # AMSGrad

        total_steps = self.epochs * (len(self.train_dataloader) // self.accum_iter)
        if self.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                             int(self.warmup_ratio * total_steps),
                                             total_steps)
        elif self.scheduler_type == "constant":
            scheduler = get_constant_schedule(self.optimizer)

        for epoch_i in range(0, self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()
            self.model.train()
            self.total_train_loss = 0.0
            
            # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
            # batch accumulation parameter
            # self.accum_iter = 1
            
            for step, batch in tqdm(enumerate(self.train_dataloader)):
                # Progress update every 40 batches.
                if step % (40 * self.accum_iter) == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step // self.accum_iter, len(self.train_dataloader) // self.accum_iter, elapsed))
                    
                logits, loss = self.model(batch[0].to(self.cuda), batch[1].to(self.cuda), batch[2], batch[3], batch[4], batch[5])
                
                #Don't update the model with loss_eo, since this is not the learning objective
                #logits_eo, loss_eo = self.model(batch[0].to(self.cuda), batch[6].to(self.cuda), batch[2], batch[3], batch[4], batch[5]) # Updated on May 17, 2022
                self.total_train_loss += loss.item()
                
                # normalize loss to account for batch accumulation
                loss = loss / self.accum_iter 
                
                # backward pass
                loss.backward()
                
                # weights update
                if ((step + 1) % self.accum_iter == 0) or (step + 1 == len(self.train_dataloader)):
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)
            print("")
            print("  Total training loss: {0:.2f}".format(self.total_train_loss))
            print("  Training epoch took: {:}".format(training_time))
            
            micro_f1, macro_f1 = self.evaluate_simple(self.valid_dataloader)
            
            log_dict = {"micro_f1":micro_f1, "macro_f1":macro_f1 } 
            
            if len(additional_eval_loader_dict) > 0:
                for name, eval_loader in additional_eval_loader_dict.items():
                    micro_f1, macro_f1 = self.evaluate_simple(eval_loader)
                    log_dict.update({f"{name}_micro_f1":micro_f1, f"{name}_macro_f1":macro_f1 })
            for key, val in log_dict.items():
                print(key, val)
            wandb.log(log_dict)

            if log_dict[self.metric] > self.best_f1 :
                self.best_f1 = log_dict[self.metric]
                for name, score in log_dict.items():
                    log_eval_best["best_"+name] = score
                # save temporary model
                torch.save(self.model.state_dict(), self.best_PATH)

        wandb.log(log_eval_best)

        print("")
        print("======== Training complete! ========")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    def evaluate_simple(self, eval_dataloader):
        y_logits = []
        y_gold = []
        y_pred = []

        for batch in eval_dataloader:
            with torch.no_grad():
                logits, loss = self.model(batch[0].to(self.cuda), batch[1].to(self.cuda), batch[2], batch[3], batch[4], batch[5])
            # Move logits and labels to CPU
            y_predict = torch.max(logits, 1).indices.cpu().numpy()
            y_pred.extend(y_predict)
            labels = []
            for batch_label in batch[5]:
                for label in batch_label:
                    labels.append(label)
            y_gold.extend(labels)

            y_logits.extend(logits.cpu().numpy())
        all_losses = F.cross_entropy(torch.tensor(y_logits), torch.tensor(y_gold), reduction="none")
        micro_f1 = f1_score(y_gold, y_pred, average='micro')
        macro_f1 = f1_score(y_gold, y_pred, average='macro')
        return micro_f1, macro_f1
            
    def evaluate(self, eval_data, test = False, predict = False, f1_metric = 'macro'):
        # ========================================
        #             Validation / Test
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        # Also applicable to test set.
        # Return 1 if the evaluation of this epoch achieves new best results,
        # else return 0.
        t0 = time.time()
            
        if test:
            if self.load_model_path:
                self.model = torch.load(self.load_model_path + self.model_name + ".pt")
            else:
                print("NOT LOADING ANY MODEL...")
                
            self.model.to(self.cuda)
            print("")
            print("loaded " + eval_data + " best model:" + self.model_name + ".pt")
            #if predict == False:
                #print("(from epoch " + str(self.best_epoch) + " )")
            print("(from epoch " + str(self.best_epoch) + " )")
            print("Running Evaluation on " + eval_data + " Test Set...")
            dataloader = self.test_dataloader
        else:
            # Evaluation
            print("")
            print("Running Evaluation on Validation Set...")
            dataloader = self.valid_dataloader
            
        self.model.eval()
        
        y_pred = []
        y_gold = []
        if self.out_class == 3:
            y_logits = np.array([[0.0, 1.0, 2.0]])
        else:
            y_logits = np.array([[0.0, 1.0, 2.0, 3.0]])
        
        # Evaluate for one epoch.
        for batch in dataloader:
            with torch.no_grad():
                logits, loss = self.model(batch[0].to(self.cuda), batch[1].to(self.cuda), batch[2], batch[3], batch[4], batch[5])
                if self.dpn:
                    logits, loss = self.model(batch[0].to(self.cuda), batch[1].to(self.cuda), batch[2], batch[3], batch[4], batch[5])
                    logits_eo, loss_eo = self.model(batch[0].to(self.cuda), batch[6].to(self.cuda), batch[2], batch[3], batch[4], batch[5]) # Updated on May 17, 2022
                    logits_xb, loss_xb = self.model(batch[0].to(self.cuda), batch[7].to(self.cuda), batch[2], batch[3], batch[4], batch[5]) # Updated on Jun 14, 2022
                    logits = nn.Softmax(dim=1)(logits) - torch.tensor(self.lambda_1) * nn.Softmax(dim=1)(logits_eo) - torch.tensor(self.lambda_2) * nn.Softmax(dim=1)(logits_xb) # Updated on Jun 14, 2022

            # Move logits and labels to CPU
            y_predict = torch.max(logits[:, 0:self.out_class], 1).indices.cpu().numpy()
            y_pred.extend(y_predict)
            
            labels = []
            for batch_label in batch[5]:
                for label in batch_label:
                    labels.append(label)
            y_gold.extend(labels)
            
            y_logits = np.append(y_logits, logits[:, 0:self.out_class].cpu().numpy(), 0) # for prediction result output # 3 if DPN; else 4
            
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("Eval took: {:}".format(validation_time))
        
        # Output prediction results.
        if predict:
            if predict[-4:] == "json":
                with open(predict, 'w') as outfile:
                    numpyData = {"labels": "0 -- Parent-Child or Before; 1 -- Child-Parent or After; 2 -- Coref or Simultaneous; 3 -- NoRel or Vague", "array": y_logits}
                    json.dump(numpyData, outfile, cls=NumpyArrayEncoder)
                #try:
                #    msg = message(subject=eval_data + " Prediction Notice",
                #                  text=self.dataset + "/" + self.model_name + " Predicted " + str(y_logits.shape[0] - 1) + " instances. (Current Path: " + os.getcwd() + ")")
                #    send(msg)  # and send it
                #except:
                #    print("Send failed.")
                #return 0
            else:
                with open(predict + "gold", 'w') as outfile:
                    for i in y_gold:
                        print(i, file = outfile)
                with open(predict + "pred", 'w') as outfile:
                    for i in y_pred:
                        print(i, file = outfile)   
        
        # Calculate the performance.
        
        if eval_data == "MATRES":
            try:  
                if self.dpn:
                    tri_gold = []
                    tri_pred = []
                    for i, label in enumerate(y_gold):
                        if label != 3:
                            tri_gold.append(label)
                            tri_pred.append(y_pred[i])
                    macro_f1 = f1_score(tri_gold, tri_pred, average='macro')
                    micro_f1 = f1_score(tri_gold, tri_pred, average='micro')
                    print("  macro F1: {0:.3f}".format(macro_f1))
                    print("  micro F1: {0:.3f}".format(micro_f1))
                    CM = confusion_matrix(tri_gold, tri_pred)
                    print(CM)
                    F1 = (micro_f1, macro_f1)
                else:
                    Acc, P, R, F1, CM = metric(y_gold, y_pred)
                    print("  P: {0:.3f}".format(P))
                    print("  R: {0:.3f}".format(R))
                    print("  F1: {0:.3f}".format(F1))
                    macro_f1 = f1_score(y_gold, y_pred, average='macro')
                    micro_f1 = f1_score(y_gold, y_pred, average='micro')
                    print("  macro f-score: {0:.3f}".format(macro_f1))
                    print("  micro f-score: {0:.3f}".format(micro_f1))
                    print(CM)
                    F1 = (micro_f1, macro_f1)

            except:
                print("No classification_report for this epoch of evaluation (Recall and F-score are ill-defined and being set to 0.0 due to no true samples).")
                
        return 0, F1
    


