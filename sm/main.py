import argparse
import logging
import os
import pickle
import pprint
import time

import numpy as np
import yaml

import compute
import dataset
import models
import settings
import torch
import torch.nn as nn
import torch.optim as optim
import utils
# from IPython.core.debugger import Pdb


def main(args):

    # Store name of experiment
    exp_name = args.exp_name
    exp_name = '{}_r{}_p{}_n{}_i{}_k{}'.format(
        exp_name, args.rho, args.pos_reward, args.neg_reward, args.class_imbalance, args.kldiv_lambda)

    # Create an directory for output path
    args.output_path = os.path.join(args.output_path, args.exp_name)
    os.makedirs(args.output_path, exist_ok=True)

    utils.LOG_FILE = os.path.join(args.output_path, 'log.txt')

    LEARNING_PROFILE_FILE = os.path.join(
        args.output_path, 'learning_curve.txt')
    lpf = open(LEARNING_PROFILE_FILE, 'a')
    args.lpf = lpf
    # Set logging
    logging.basicConfig(filename=utils.LOG_FILE, filemode='a', format='%(levelname)s :: %(asctime)s - %(message)s',
                        level=args.log_level, datefmt='%d/%m/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setLevel(args.log_level)
    formatter = logging.Formatter(
        '%(levelname)s :: %(asctime)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logging.info('Beginning code for experiment {} and storing stuff in {}'.format(
        exp_name, args.output_path))
    logging.info('Loaded arguments as \n{}'.format(str(pprint.pformat(args))))



    # Begin of main code

    train_loader, val_loader, labelled_train_loader = dataset.get_data_loaders(
        args)
    model = models.select_model(args)
    my_eval_fn = compute.get_evaluation_function(args)

    if args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
        )), momentum=args.momentum, lr=args.lr, weight_decay=args.decay)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters(
        )), lr=args.lr, weight_decay=args.decay)

    checkpoint_file = os.path.join(
        args.output_path, '{}_checkpoint.pth'.format(exp_name))
    best_checkpoint_file = os.path.join(
        args.output_path, '{}_best_checkpoint.pth'.format(exp_name))
    logging.info('Saving checkpoints at {} and best checkpoint at : {}'.format(
        checkpoint_file, best_checkpoint_file))

    start_epoch = 0
    best_score = -9999999

    # Load checkpoint if present in input arguments
    if args.checkpoint != '':
        logging.info('Starting from checkpoint: {}'.format(args.checkpoint))
        cp = torch.load(args.checkpoint)
        start_epoch = cp['epoch'] + 1
        model.load_state_dict(cp['model'])
        # optimizer.load_state_dict(cp['optimizer']) TODO: - Why not do this?
        best_score = cp['best_score']
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            param_group['weight_decay'] = args.decay

    num_epochs = args.num_epochs
    logging.info('Beginning train/validate cycle')

    time1 = time.time()
    if val_loader is not None:
        record, metric_idx, headers = compute.compute(start_epoch-1, model, val_loader, optimizer,
                                                      'eval', eval_fn=my_eval_fn, args=args)
        if(args.log_eval is not None):
            handler = open(args.log_eval, "a")
            print(','.join([str(round(x, 6)) if isinstance(
                x, float) else str(x) for x in record]), file=handler)
            handler.close()
    print("Time taken:",time.time()-time1)
    if(args.only_eval):
        logging.info('Ran only eval mode, now exiting')
        exit(0)

    # Start TRAINING
    for epoch in range(start_epoch, num_epochs):
        logging.info('Beginning epoch {}'.format(epoch))

        if labelled_train_loader is not None:
            record, metric_idx, _ = compute.compute(
                epoch, model, labelled_train_loader, optimizer, 'train_sup', eval_fn=my_eval_fn, args=args)

        if train_loader is not None:
            record, metric_idx, _ = compute.compute(
                epoch, model, train_loader, optimizer, 'train_un', eval_fn=my_eval_fn, args=args, labelled_train_loader=labelled_train_loader)

        if val_loader is not None:
            record, metric_idx, _ = compute.compute(
                epoch, model, val_loader, None, 'eval', eval_fn=my_eval_fn, args=args)

        is_best = False
        logging.info('Best score: {}, This score: {}'.format(
            best_score, record[metric_idx]))

        if record[metric_idx] > best_score:
            best_score = record[metric_idx]
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'best_score': best_score,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'is_best': is_best
        }, epoch, is_best, checkpoint_file, best_checkpoint_file)

    args.lpf.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path',
                        help="Training data path (pkl file)", type=str)

    parser.add_argument('--labelled_training_data_path',
                        help="Labelled Training data path (pkl file)", type=str)

    parser.add_argument('--base_model_file',
                        help="Base model dump for loading embeddings", type=str, default='')
    parser.add_argument(
        '--val_data_path', help="Validation data path in the same format as training data", type=str, default='')
    parser.add_argument(
        '--val_labels_path', help="Validation data Labels path for multi-label evaluation", type=str, default=None)

    parser.add_argument(
        '--train_labels_path', help="Training data Labels path for multi-label training", type=str, default=None)

    parser.add_argument('--exp_name', help='Experiment name',
                        type=str, default='default_exp')
    parser.add_argument(
        '--output_path', help='Output path to store models, and logs', type=str, required=True)

    # Training parameters
    parser.add_argument('--num_epochs', help='epochs', type=int, default=100)
    parser.add_argument(
        '--log_after', help='Log after samples', type=int, default=200000)
    parser.add_argument('--batch_size', help='batch size',
                        type=int, default=256)

    # Model parameters
    parser.add_argument('--each_input_size',
                        help='Input size of each template', type=int, default=7)
    parser.add_argument(
        '--num_templates', help='number of templates excluding other', type=int, default=6)
    parser.add_argument('--use_ids', help='Use embeddings of entity and relations while training',
                        action='store_true', default=False)
    parser.add_argument('--mil', help='Use MIL model',
                        action='store_true', default=False)
    parser.add_argument('--exclude_t_ids', nargs='*', type=int, required=False,default=[], help='List of templates to be excluded while making predictions')
    
    parser.add_argument('--hidden_unit_list', nargs='*', type=int, required=False,default=[], help='number of hidden neurons in each layer')

    # Optimizer parameters
    parser.add_argument(
        '--optim', help='type of optimizer to use: sgd or adam', type=str, default='sgd')
    parser.add_argument('--lr', help='lr', type=float, default=0.001)
    parser.add_argument('--decay', help='lr', type=float, default=0)
    parser.add_argument('--momentum', help='lr', type=float, default=0.9)

    parser.add_argument(
        '--checkpoint', help='checkpoint path for loading model', type=str, default='')
    parser.add_argument('--config', help='yaml config file',
                        type=str, default='default_config.yml')
    parser.add_argument('--cuda', help='if cuda available, use it or not?',
                        action='store_true', default=False)

    parser.add_argument('--only_eval', help='Only evaluate?',
                        action='store_true', default=False)
    parser.add_argument(
        '--log_eval', help="logs eval accuracies", default=None, type=str)

    parser.add_argument(
        '--pred_file', help="predictions log file", default=None, type=str)

    parser.add_argument('--log_level', help='Set the logging output level. {0}'.format(
        utils._LOG_LEVEL_STRINGS), default='INFO', type=utils._log_level_string_to_int, nargs='?')

    parser.add_argument('--supervision', help='possible values - un, semi, sup',
                        type=str, default='un')

    #kl div loss args
    parser.add_argument('--kldiv_lambda', help='relative wt of kl div loss', default=0.0,type=float)
    parser.add_argument('--label_distribution_file',help='yaml file containing target distribution', default= 'label_distrubution.yml' , type=str)

    #other loss hyperparameters:
    parser.add_argument('--neg_reward', help='negative reward', default=-1, type=float)
    parser.add_argument('--rho', help='rho ', default=0.125, type=float)

    #multi label train/eval is done when: eval_labels_file is not None and eval_ml = 1. Hence default behaviour is: single label as default value of eval_labels_file is None
    parser.add_argument('--train_ml',help='should use multi label loss?', default = 1, type=int)
    parser.add_argument('--eval_ml',help='should eval multi label ?', default = 1, type=int)
    
    parser.add_argument('--default_value',help='default value of template score when it is undefined?', default = 0, type=float)
    parser.add_argument('--exclude_default',help='should default value be excluded while computing stats?', default = 0, type=int)


    args = parser.parse_args()
    config = {}
    if os.path.exists(os.path.expanduser(args.config)):
        config = yaml.load(open(os.path.expanduser(args.config)))
    config.update(vars(args))
    config.update({'embed_size': utils.get_embed_size(
        args.base_model_file, args.use_ids)})
    args = utils.Map(config)
    o2n , n2o = utils.get_template_id_maps(args.num_templates, args.exclude_t_ids)
    args.o2n = o2n
    args.n2o = n2o

    for key in ['train_labels_path','val_labels_path']:
        if args[key]  == 'None':
            args[key] = None

    
    settings.set_settings(args)
    main(args)
