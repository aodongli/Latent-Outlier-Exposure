""" This code is shared for review purposes only. Do not copy, reproduce, share, publish,
or use for any purpose except to review our ICML submission. Please delete after the review process.
The authors plan to publish the code deanonymized and with a proper license upon publication of the paper. """

import os
import json
import torch
import random
import numpy as np
from loader.LoadData import load_data
from utils import Logger


class KVariantEval:
    """
    Class implementing a sufficiently general framework to do model ASSESSMENT
    """

    def __init__(self, dataset, exp_path, model_configs,true_contamination,est_contamination):
        self.num_cls = dataset.num_cls
        self.data_name = dataset.data_name
        self.data_contamination = true_contamination
        self.model_contamination = est_contamination
        self.model_configs = model_configs
        self.exp_path = exp_path
        self._NESTED_FOLDER = os.path.join(exp_path, str(true_contamination)+'_'+str(est_contamination))
        self._FOLD_BASE = '_CLS'
        self._RESULTS_FILENAME = 'results.json'
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def process_results(self):

        TS_f1s = []
        TS_aps = []
        TS_aucs = []

        assessment_results = {}

        for i in range(self.num_cls):
            try:
                config_filename = os.path.join(self._NESTED_FOLDER, str(i)+self._FOLD_BASE,
                                               self._RESULTS_FILENAME)

                with open(config_filename, 'r') as fp:
                    variant_scores = json.load(fp)
                    ts_f1 = np.array(variant_scores['TS_F1'])
                    ts_auc = np.array(variant_scores['TS_AUC'])
                    ts_ap = np.array(variant_scores['TS_AP'])
                    TS_f1s.append(ts_f1)
                    TS_aucs.append(ts_auc)
                    TS_aps.append(ts_ap)

                assessment_results['avg_TS_f1_' + str(i)] = ts_f1.mean()
                assessment_results['std_TS_f1_' + str(i)] = ts_f1.std()
                assessment_results['avg_TS_ap_' + str(i)] = ts_ap.mean()
                assessment_results['std_TS_ap_' + str(i)] = ts_ap.std()
                assessment_results['avg_TS_auc_' + str(i)] = ts_auc.mean()
                assessment_results['std_TS_auc_' + str(i)] = ts_auc.std()
            except Exception as e:
                print(e)

        TS_f1s = np.array(TS_f1s)
        TS_aps = np.array(TS_aps)
        TS_aucs = np.array(TS_aucs)
        avg_TS_f1 = np.mean(TS_f1s, 0)
        avg_TS_ap = np.mean(TS_aps, 0)
        avg_TS_auc = np.mean(TS_aucs, 0)
        assessment_results['avg_TS_f1_all'] = avg_TS_f1.mean()
        assessment_results['std_TS_f1_all'] = avg_TS_f1.std()
        assessment_results['avg_TS_ap_all'] = avg_TS_ap.mean()
        assessment_results['std_TS_ap_all'] = avg_TS_ap.std()
        assessment_results['avg_TS_auc_all'] = avg_TS_auc.mean()
        assessment_results['std_TS_auc_all'] = avg_TS_auc.std()

        with open(os.path.join(self._NESTED_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump(assessment_results, fp)

    def risk_assessment(self, experiment_class,oe_est):
        """
        :param experiment_class: the kind of experiment used
        :param debug:
        :param other: anything you want to share across processes
        :return: An average over the outer test folds. RETURNS AN ESTIMATE, NOT A MODEL!!!
        """
        self.oe_est = oe_est
        if not os.path.exists(self._NESTED_FOLDER):
            os.makedirs(self._NESTED_FOLDER)

        for cls in range(self.num_cls):

            # Create a separate folder for each experiment
            folder = os.path.join(self._NESTED_FOLDER, str(cls)+self._FOLD_BASE)
            if not os.path.exists(folder):
                os.makedirs(folder)

            json_results = os.path.join(folder, self._RESULTS_FILENAME)
            if not os.path.exists(json_results):

                self._risk_assessment_helper(cls,  experiment_class, folder)
            else:
                # Do not recompute experiments for this outer fold.
                print(
                    f"File {json_results} already present! Shutting down to prevent loss of previous experiments")
                continue

        self.process_results()

    def _risk_assessment_helper(self, cls, experiment_class, exp_path):

        best_config = self.model_configs[0]
        experiment = experiment_class(best_config, exp_path,self.oe_est)

        # Set up a log file for this experiment (run in a separate process)

        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')
        # logger = None

        val_auc_list, test_auc_list,test_ap_list,test_f1_list = [], [],[], []
        num_repeat = best_config['num_repeat']
        # Mitigate bad random initializations
        for i in range(num_repeat):
            torch.cuda.empty_cache()
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(i + 40)
            random.seed(i + 40)
            torch.manual_seed(i + 40)
            torch.cuda.manual_seed(i + 40)
            torch.cuda.manual_seed_all(i + 40)
            trainset, valset, testset = load_data(self.data_name, cls, self.data_contamination)
            val_auc, test_auc, test_ap,test_f1, test_score = experiment.run_test(trainset,valset,testset,logger,self.model_contamination)
            print(f'Final training run {i + 1}: {val_auc}, {test_auc,test_ap, test_f1}')

            val_auc_list.append(val_auc)
            test_auc_list.append(test_auc)
            test_ap_list.append(test_ap)
            test_f1_list.append(test_f1)


        if logger is not None:
            logger.log(
                'End of Variant:'+ str(cls) + ' TS f1: ' + str(test_f1_list)+' TS AP: ' + str(test_ap_list)+' TS auc: ' + str(test_auc_list) )
        print('F1:'+str(np.array(test_f1_list).mean())+'AUC:'+str(np.array(test_auc_list).mean()))
        with open(os.path.join(exp_path, self._RESULTS_FILENAME), 'w') as fp:
            json.dump({'best_config': best_config, 'VAL_AUC': val_auc_list,
                       'TS_F1': test_f1_list,'TS_AP': test_ap_list,'TS_AUC': test_auc_list}, fp)


