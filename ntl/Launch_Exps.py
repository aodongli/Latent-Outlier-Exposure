""" This code is shared for review purposes only. Do not copy, reproduce, share, publish,
or use for any purpose except to review our ICML submission. Please delete after the review process.
The authors plan to publish the code deanonymized and with a proper license upon publication of the paper. """

import argparse
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from config.base import Grid, Config
from evaluation.Experiments import runExperiment
from evaluation.Kvariants_Eval import KVariantEval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_thyroid.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='thyroid')
    parser.add_argument('--true_contamination', type=float, default=0.1)
    parser.add_argument('--est_contamination', type=float, default=0.1)
    parser.add_argument('--oe_est', default='hard') # blind, refine, hard, soft
    return parser.parse_args()

def EndtoEnd_Experiments(config_file, dataset_name,true_contamination,est_contamination,oe_est):

    model_configurations = Grid(config_file, dataset_name)
    model_configuration = Config(**model_configurations[0])
    dataset =model_configuration.dataset
    result_folder = model_configuration.result_folder+model_configuration.exp_name
    exp_path = os.path.join(result_folder,f'{oe_est}')


    risk_assesser = KVariantEval(dataset, exp_path, model_configurations,true_contamination,est_contamination)

    risk_assesser.risk_assessment(runExperiment,oe_est)

if __name__ == "__main__":
    args = get_args()

    if args.oe_est != 'all':
        oe_est = [args.oe_est]
    else:
        oe_est = ['soft','hard','refine','blind']


    config_file = 'config_files/'+args.config_file

    for oe_type in oe_est:
        try:
            EndtoEnd_Experiments(config_file, args.dataset_name,args.true_contamination,args.est_contamination,oe_type)

        except Exception as e:
            raise e  # print(e)