""" This code is shared for review purposes only. Do not copy, reproduce, share,
publish, or use for any purpose except to review our ICML submission. Please
delete after the review process. The authors plan to publish the code
deanonymized and with a proper license upon publication of the paper. """

import torch.utils.data
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.backends import cudnn
from wideresnet import MultiHeadWideResNet
from sklearn.metrics import roc_auc_score, average_precision_score

import logging

cudnn.benchmark = True

def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


class TransClassifier():
    def __init__(self, total_num_trans, num_trans_list, args):
        self.n_trans = total_num_trans
        self.n_trans_list = num_trans_list
        self.n_heads = len(self.n_trans_list)

        self.args = args
        if args.dataset == 'cifar10':
            self.n_channels = 3
        elif args.dataset == 'fmnist':
            self.n_channels = 1
        self.netWRN = MultiHeadWideResNet(
                                 self.args.depth, num_trans_list, 
                                 n_channels=self.n_channels, 
                                 widen_factor=self.args.widen_factor,
                                 dropRate=0.3).cuda()
        self.optimizer = torch.optim.Adam(self.netWRN.parameters(),
                                          lr=self.args.lr)

        self.__oe_loss__ = self.args.oe_loss

    def fit_trans_classifier(self, x_train, mt_train_labels, y_train, x_test, mt_test_labels, y_test):
        #////// Nesterov SGD optimizer //////
        if self.args.sgd_opt:
            self.optimizer = torch.optim.SGD(self.netWRN.parameters(),
                                             lr=self.args.lr,
                                             momentum=0.9,
                                             weight_decay=5e-4,
                                             nesterov=True)
        # ////////////////////////////////////

        if self.args.lr_decay:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: cosine_annealing(
                    step,
                    self.args.epochs * len(list(range(0, len(x_train), self.args.batch_size))),
                    1,  # since lr_lambda computes multiplicative factor
                    1e-6 / self.args.lr))

        if self.args.epoch_lr_decay:
            self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10, gamma=0.1)

        print("Training")
        self.netWRN.train()
        bs = self.args.batch_size
        N, sh, sw, nc = x_train.shape
        n_rots = self.n_trans
        m = self.args.m
        celoss = torch.nn.CrossEntropyLoss(reduction='none')
        ndf = 256

        aucs = np.zeros(self.args.epochs)
        aucs_latent_gauss = np.zeros(self.args.epochs)

        for epoch in range(self.args.epochs):
            self.netWRN.train()
            total_loss = 0.0
            update_num = 0
            
            rp = np.random.permutation(N//n_rots)
            rp = np.concatenate([np.arange(n_rots) + rp[i]*n_rots for i in range(len(rp))])
            assert len(rp) == N
            all_zs = torch.zeros((len(x_train), ndf)).cuda()
            diffs_all = []
            all_logp = torch.zeros((N//n_rots, n_rots, n_rots)).cuda()
            for i in range(0, len(x_train), bs):
                batch_range = min(bs, len(x_train) - i)
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(x_train[rp[idx]]).float().cuda()
                gt_labels = torch.from_numpy(y_train[rp[idx]]).long().cuda()
                mt_labels = torch.from_numpy(mt_train_labels[rp[idx]]).long().cuda()

                zs_tc, mt_zs_ce = self.netWRN(xs)

                all_zs[idx] = zs_tc
                train_labels = torch.from_numpy(np.tile(np.arange(n_rots), batch_range//n_rots)).long().cuda()
                zs = torch.reshape(zs_tc, (batch_range//n_rots, n_rots, ndf))

                pos_loss = 0
                for t_ind, zs_ce in enumerate(mt_zs_ce):
                    # for each head
                    pos_loss += celoss(zs_ce, mt_labels[:, t_ind])
                pos_loss /= self.n_heads

                if self.args.oe:

                    # rank against anomaly scores

                    # # ranking: training loss anomaly score 
                    if self.args.oe_rank == 'training_obj':
                        logp_sz = 0.
                        for t_ind, zs_ce in enumerate(mt_zs_ce):
                            logp_sz_tot = -F.log_softmax(self.args.ad_temp*zs_ce, dim=1)
                            logp_sz_t = logp_sz_tot[np.arange(batch_range), mt_labels[:, t_ind]]
                            logp_sz += torch.reshape(logp_sz_t, (batch_range // n_rots, n_rots)).sum(1) # (36,)
                        logp_sz /= self.n_heads

                    # ranking: latent gaussian anomaly score 
                    elif self.args.oe_rank == 'latent_gauss':
                        means = zs.mean(0).unsqueeze(0).detach()
                        diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)
                        diffs_eps = self.args.eps * torch.ones_like(diffs)
                        diffs = torch.max(diffs, diffs_eps)
                        logp_sz = torch.nn.functional.log_softmax(-self.args.ad_temp*diffs, dim=2)
                        # logp_sz = -torch.diagonal(logp_sz, dim1=1, dim2=2).reshape(batch_range).detach()
                        logp_sz = -torch.diagonal(logp_sz, dim1=1, dim2=2).sum(1).detach()

                    else:
                        raise NotImplementedError


                    neg_loss = 0
                    for t_ind, zs_ce in enumerate(mt_zs_ce):
                        # for each head
                        t_classes = self.n_trans_list[t_ind]
                        pseudo_train_labels = torch.ones((batch_range, t_classes)).to(zs_tc)/t_classes
                        neg_loss -= (F.log_softmax(zs_ce, dim=1) * pseudo_train_labels).sum(1)
                    neg_loss /= self.n_heads
                    neg_loss = torch.reshape(neg_loss, (batch_range // n_rots, n_rots)).sum(1)

                    logp_sz -= neg_loss


                    loss_accept, idx_accept = torch.topk(logp_sz, int(logp_sz.shape[0] * 0.9), largest=False, sorted=False)
                    loss_reject, idx_reject = torch.topk(logp_sz, int(logp_sz.shape[0] * 0.1), largest=True, sorted=False)

                    idx_accept = (torch.arange(n_rots).to(zs_ce).repeat(idx_accept.size()[0]) + idx_accept.repeat_interleave(n_rots)*n_rots).long()
                    idx_reject = (torch.arange(n_rots).to(zs_ce).repeat(idx_reject.size()[0]) + idx_reject.repeat_interleave(n_rots)*n_rots).long()
                    

                    if self.args.oe_method == 'zero_driven':

                        # negative loss 
                        neg_loss = 0.
                        for t_ind, zs_ce in enumerate(mt_zs_ce):
                            prob = F.softmax(self.args.ad_temp*zs_ce, dim=1)
                            prob_t = prob[np.arange(batch_range), mt_labels[:, t_ind]]
                            prob_t[prob_t < 1e-9] = 1e-9 # prevents NaN
                            neg_loss_t = -torch.log(prob_t)
                            neg_loss += neg_loss_t
                        neg_loss /= self.n_heads

                    elif self.args.oe_method == 'max_ent':
                        # opt 2 for potential outliers: CE between uniform
                        # distribution and negative TC

                        neg_loss = 0
                        for t_ind, zs_ce in enumerate(mt_zs_ce):
                            # for each head
                            t_classes = self.n_trans_list[t_ind]
                            pseudo_train_labels = torch.ones((batch_range, t_classes)).to(zs_tc)/t_classes
                            neg_loss -= (F.log_softmax(zs_ce, dim=1) * pseudo_train_labels).sum(1)
                        neg_loss /= self.n_heads

                    else:
                        raise NotImplementedError

                    if self.args.oe_loss == 'weighted':
                        _loss = torch.cat([pos_loss[idx_accept],(1-self.args.oe_weight)*pos_loss[idx_reject]+(self.args.oe_weight)*neg_loss[idx_reject]],0)
                    elif self.args.oe_loss == 'radical':
                        _loss = torch.cat([pos_loss[idx_accept], self.args.oe_weight*neg_loss[idx_reject]], 0)
                    elif self.args.oe_loss == 'refine':
                        _loss = pos_loss[idx_accept]
                    elif self.args.oe_loss == 'known_gt':
                        _loss = torch.cat([pos_loss[gt_labels==0], neg_loss[gt_labels==1]],0)
                    else:
                        raise NotImplementedError
                else:
                    _loss = pos_loss

                tc = tc_loss(zs, m)
                ce = _loss.mean() 
                if self.args.reg:
                    loss = ce + self.args.lmbda * tc + 10 *(zs*zs).mean()
                else:
                    loss = ce + self.args.lmbda * tc

                update_num += 1
                total_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.args.lr_decay:
                    self.scheduler.step()

            self.netWRN.eval()
            all_zs = torch.reshape(all_zs, (N//n_rots, n_rots, ndf))
            means = all_zs.mean(0, keepdim=True)

            # evaluation
            print("## Training Objective Anomaly Score")
            logging.info("## Training Objective Anomaly Score")
            # make the training objective equivalent to anomaly score
            val_loss = 0.0
            val_update_num = 0
            with torch.no_grad():
                batch_size = bs
                val_probs_rots_trainObj = np.zeros((len(y_test),))
                val_probs_dirichlet = np.zeros((len(y_test),))
                
                for i in range(0, len(x_test), batch_size):
                    batch_range = min(batch_size, len(x_test) - i)
                    idx = np.arange(batch_range) + i
                    xs = torch.from_numpy(x_test[idx]).float().cuda()
                    mt_labels = torch.from_numpy(mt_test_labels[idx]).long().cuda()

                    # anomaly score
                    zs_tc, mt_zs_ce = self.netWRN(xs)
                    val_probs = 0.
                    for t_ind, zs_ce in enumerate(mt_zs_ce):
                        t_classes = self.n_trans_list[t_ind]
                        logp_sz = -F.log_softmax(self.args.ad_temp*zs_ce, dim=1)
                        logp_sz_t = logp_sz[np.arange(batch_range), mt_labels[:, t_ind]]
                        val_probs += torch.reshape(logp_sz_t, (batch_range // n_rots, n_rots)).sum(1)
                    val_probs /= self.n_heads

                    zs_reidx = np.arange(batch_range // n_rots) + i // n_rots
                    val_probs_rots_trainObj[zs_reidx] = val_probs.cpu().data.numpy()

                    # validation loss
                    ce = 0
                    for t_ind, zs_ce in enumerate(mt_zs_ce):
                        ce += celoss(zs_ce, mt_labels[:, t_ind])
                    ce /= self.n_heads

                    zs = torch.reshape(zs_tc, (batch_range // n_rots, n_rots, ndf))
                    tc = tc_loss(zs, m)

                    if self.args.reg:
                        _val_loss = ce + self.args.lmbda * tc + 10 *(zs*zs).mean()
                    else:
                        _val_loss = ce + self.args.lmbda * tc

                    val_loss += _val_loss.mean().item()
                    val_update_num += 1

                aucs[epoch] = roc_auc_score(y_test, val_probs_rots_trainObj)
                ap = average_precision_score(y_test, val_probs_rots_trainObj)
                print("Epoch:", epoch, ", AUC: ", aucs[epoch], ", AP: ", ap)
                logging.info("Epoch:" + f'{epoch}' + 
                             ", AUC: " + f'{aucs[epoch]}' + 
                             ", AP: " + f'{ap}')

                total_loss /= update_num
                print("Average training loss: ", total_loss)
                logging.info(f'Average training loss: {total_loss}')

                val_loss /= val_update_num
                print("Average validation loss: ", val_loss)
                logging.info(f'Average validation loss: {val_loss}')


            print("## Latent Gaussian Anomaly Score")
            logging.info("## Latent Gaussian Anomaly Score")
            with torch.no_grad():
                batch_size = bs
                val_probs_rots_latentGauss = np.zeros((len(y_test), self.n_trans))
                val_probs_dirichlet = np.zeros((len(y_test),))
                for i in range(0, len(x_test), batch_size):
                    batch_range = min(batch_size, len(x_test) - i)
                    idx = np.arange(batch_range) + i
                    xs = torch.from_numpy(x_test[idx]).float().cuda()

                    zs, _ = self.netWRN(xs)
                    zs = torch.reshape(zs, (batch_range // n_rots, n_rots, ndf))

                    diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)
                    diffs_eps = self.args.eps * torch.ones_like(diffs)
                    diffs = torch.max(diffs, diffs_eps)
                    logp_sz = torch.nn.functional.log_softmax(-self.args.ad_temp*diffs, dim=2)

                    zs_reidx = np.arange(batch_range // n_rots) + i // n_rots
                    val_probs_rots_latentGauss[zs_reidx] = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy()

                val_probs_rots_latentGauss = val_probs_rots_latentGauss.sum(1)
                aucs_latent_gauss[epoch] = roc_auc_score(y_test, val_probs_rots_latentGauss)
                ap = average_precision_score(y_test, val_probs_rots_latentGauss)
                print("Epoch:", epoch, ", AUC: ", aucs_latent_gauss[epoch], ", AP: ", ap)
                logging.info("Epoch:" + f'{epoch}' + 
                             ", AUC: " + f'{aucs_latent_gauss[epoch]}' + 
                             ", AP: " + f'{ap}')

            print("## Ensemble Anomaly Score")
            logging.info("## Ensemble Anomaly Score")
            val_probs_rots = val_probs_rots_trainObj + val_probs_rots_latentGauss
            auc = roc_auc_score(y_test, val_probs_rots)
            ap = average_precision_score(y_test, val_probs_rots_latentGauss)
            print("Epoch:", epoch, ", AUC: ", auc, ", AP: ", ap)
            logging.info("Epoch:" + f'{epoch}' + 
                         ", AUC: " + f'{auc}' + 
                         ", AP: " + f'{ap}')

            if self.args.epoch_lr_decay:
                self.epoch_scheduler.step()

        # save training logs
        np.save(self.args._foldername + './aucs.npy', aucs)
        np.save(self.args._foldername + './aucs_latent_gauss.npy', aucs_latent_gauss)

        print(aucs[-5:])
        print(aucs_latent_gauss[-5:])

        return aucs[-1], aucs_latent_gauss[-1]

