""" This code is shared for review purposes only. Do not copy, reproduce, share, publish,
or use for any purpose except to review our ICML submission. Please delete after the review process.
The authors plan to publish the code deanonymized and with a proper license upon publication of the paper. """

import time
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
from utils import compute_pre_recall_f1,format_time
class NeutralAD_trainer:

    def __init__(self, model, loss_function,device='cuda'):

        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def _train(self, epoch,train_loader, optimizer,contamination):

        self.model.train()

        loss_all = 0
        idx = 0

        for data in train_loader:
            samples = data['sample']
            labels = data['label']
            # samples = samples.to(self.device)

            z = self.model(samples)

            loss_pos,loss_neg = self.loss_fun(z)

            if epoch <=self.warmup:

                if self.oe_est == 'gt':
                    loss = torch.cat([loss_pos[labels==0],loss_neg[labels==1]],0)
                    loss_reg = loss.mean()
                else:
                    loss = loss_pos
                    loss_reg= loss.mean()

            else:
                score = loss_pos-loss_neg

                if self.oe_est=='blind':
                    loss = loss_pos
                    loss_reg = loss.mean()
                elif self.oe_est=='hard':
                    loss_accept, idx_accept = torch.topk(score, int(score.shape[0] * (1-contamination)), largest=False,
                                                         sorted=False)
                    loss_reject, idx_reject = torch.topk(score, int(score.shape[0] * contamination), largest=True,
                                                         sorted=False)
                    loss = torch.cat([loss_pos[idx_accept], loss_neg[idx_reject]], 0)
                    loss_reg = loss.mean()
                elif self.oe_est == 'soft':
                    loss_accept, idx_accept = torch.topk(score, int(score.shape[0] * (1-contamination)), largest=False, sorted=False)
                    loss_reject, idx_reject = torch.topk(score, int(score.shape[0] * contamination), largest=True, sorted=False)
                    loss = torch.cat([loss_pos[idx_accept],0.5*loss_pos[idx_reject]+0.5*loss_neg[idx_reject]],0)
                    loss_reg= loss.mean()
                elif self.oe_est == 'refine':
                    loss_accept, idx_accept = torch.topk(loss_pos, int(loss_pos.shape[0] * (1-contamination)), largest=False,
                                                         sorted=False)
                    loss = loss_pos[idx_accept]
                    loss_reg = loss.mean()
                elif self.oe_est == 'gt':
                    loss = torch.cat([loss_pos[labels==0],loss_neg[labels==1]],0)
                    loss_reg = loss.mean()
            optimizer.zero_grad()
            loss_reg.backward()
            optimizer.step()

            loss_all += loss.sum()
            idx+=1

        return loss_all.item()/len(train_loader.dataset)


    def detect_outliers(self, loader):
        model = self.model
        model.eval()

        loss_in = 0
        loss_out = 0
        target_all = []
        score_all = []
        for data in loader:
            with torch.no_grad():
                samples = data['sample']
                labels = data['label']
                # masks = data['mask']
                # samples = samples.to(self.device)
                z= model(samples)
                loss_pos,loss_neg = self.loss_fun(z)
                score = loss_pos
                loss_in += loss_pos[labels == 0].sum()
                loss_out += loss_pos[labels == 1].sum()
                target_all.append(labels)

                score_all.append(score)

        try:
            score_all = np.concatenate(score_all)
        except:
            score_all = torch.cat(score_all).cpu().numpy()
        target_all = np.concatenate(target_all)
        auc = roc_auc_score(target_all, score_all)
        f1 = compute_pre_recall_f1(target_all,score_all)
        # f1 = 0
        ap = average_precision_score(target_all, score_all)
        return auc, ap,f1,  score_all,loss_in.item() / (target_all == 0).sum(), loss_out.item() / (target_all == 1).sum()


    def train(self, train_loader,max_epochs=100, warmup_epoch = 1,oe_est=None,contamination=0.0,optimizer=None, scheduler=None,
              validation_loader=None, test_loader=None, early_stopping=None, logger=None, log_every=10):
        self.oe_est = oe_est

        self.warmup = warmup_epoch
        early_stopper = early_stopping() if early_stopping is not None else None

        val_auc, val_f1, = -1, -1
        test_auc, test_f1, test_score = None, None,None,

        time_per_epoch = []

        if max_epochs==0:
            if test_loader is not None:
                test_auc, test_ap,test_f1,  test_score,_= self.detect_outliers(test_loader)

            if validation_loader is not None:
                val_auc, val_ap,val_f1, _,_ = self.detect_outliers(validation_loader)
            return val_auc, val_f1,test_auc,test_f1,test_score
        else:

            for epoch in range(1, max_epochs+1):

                start = time.time()
                train_loss = self._train(epoch,train_loader, optimizer,contamination)

                end = time.time() - start
                time_per_epoch.append(end)

                if scheduler is not None:
                    scheduler.step()

                if test_loader is not None:
                    test_auc, test_ap,test_f1, test_score, testin_loss,testout_loss = self.detect_outliers(test_loader)

                if validation_loader is not None:
                    val_auc, val_ap,val_f1, _, valin_loss,valout_loss = self.detect_outliers(validation_loader)
                    if epoch>10:
                        if early_stopper is not None and early_stopper.stop(epoch, valin_loss, val_auc, testin_loss, test_auc, test_ap,test_f1,
                                                                            test_score,
                                                                            train_loss):
                            break

                if epoch % log_every == 0 or epoch == 1:
                    msg = f'Epoch: {epoch}, TR loss: {train_loss}, VAL loss: {valin_loss,valout_loss}, VL auc: {val_auc} VL ap: {val_ap} VL f1: {val_f1} '

                    if logger is not None:
                        logger.log(msg)
                        print(msg)
                    else:
                        print(msg)

            if early_stopper is not None:
                train_loss, val_loss, val_auc, test_loss, test_auc, test_ap, test_f1, test_score, best_epoch \
                    = early_stopper.get_best_vl_metrics()
                msg = f'Stopping at epoch {best_epoch}, TR loss: {train_loss}, VAL loss: {val_loss}, VAL auc: {val_auc} ,' \
                    f'TS loss: {test_loss}, TS auc: {test_auc} TS ap: {test_ap} TS f1: {test_f1}'
                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)

            time_per_epoch = torch.tensor(time_per_epoch)
            avg_time_per_epoch = float(time_per_epoch.mean())

            elapsed = format_time(avg_time_per_epoch)

            return val_loss, val_auc, test_auc, test_ap,test_f1, test_score