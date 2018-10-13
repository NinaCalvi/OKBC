
import torch.optim as optim
import yaml
import torch.nn as nn
import os
import time
import argparse
import pickle
import torch
from IPython.core.debugger import Pdb
from torch.utils.data import Dataset, DataLoader
import utils
import dataset
import models
import numpy as np
import compute
import settings


def complete_paths(config):
    for key in []:
        config[key] = os.path.expanduser(config[key])
    #

def main(args):


    utils.log('done loading vocab')

    args.input_size = 15
    args.output_size = 5
    args.hidden_unit_list = [10,20]

    (train_loader, val_loader) = dataset.get_data_loaders(args, vocabs)

    model = models.select_model(args)
    
    my_eval_fn = compute.get_evaluation_function(args)

    # TODO - other optimizers?
    optimizer = {}
    if args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
        )), momentum=args.momentum, lr=args.lr, weight_decay=args.decay)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                               weight_decay=args.decay)

    my_lr_scheduler = scheduler.CustomReduceLROnPlateau(optimizer, {'mode': args.mode, 'factor': args.factor, 'patience': args.patience, 'verbose': True, 'threshold': args.threshold,
                                                                    'threshold_mode': args.threshold_mode, 'cooldown': args.cooldown, 'min_lr': args.min_lr, 'eps': args.eps}, maxPatienceToStopTraining=args.max_patience)

    exp_name = args.exp_name
    if args.debug:
        exp_name = 'd'+args.exp_name

    args.output_path = os.path.join(args.output_path, args.exp_name)
    if not os.path.exists(args.output_path):
        try:
            os.makedirs(args.output_path)
        except:
            # TODO - why pass? Why not exit? Where will we store the results?
            pass

    utils.CONSOLE_FILE = os.path.join(args.output_path, 'IPYTHON_CONSOLE')
    utils.log(str(args))

    log_file = '{}.csv'.format(exp_name)
    checkpoint_file = os.path.join(
        args.output_path, '{}_checkpoint.pth'.format(exp_name))
    best_checkpoint_file = os.path.join(
        args.output_path, '{}_best_checkpoint.pth'.format(exp_name))
    utils.log('save checkpoints at {} and best checkpoint at : {}'.format(
        checkpoint_file, best_checkpoint_file))

    tfh = open(os.path.join(args.output_path, log_file), 'a')
    start_epoch = 0
    best_score = -9999999
    # Load checkpoint if present in input arguments TODO - be careful so as not to overwrite any checkpoints
    if args.checkpoint != '':
        utils.log('start from checkpoint: {}'.format(args.checkpoint))
        load_checkpoint_file = args.checkpoint
        cp = torch.load(os.path.join(args.output_path, args.checkpoint))
        start_epoch = cp['epoch'] + 1
        model.load_state_dict(cp['model'])
        # optimizer.load_state_dict(cp['optimizer']) TODO - Why not do this?
        best_score = cp['best_score']
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            param_group['weight_decay'] = args.decay

    num_epochs = args.num_epochs
    # Pdb().set_trace()
    utils.log('start train/validate cycle')
    if args.debug:
        pass
        #val_loader = train_loader



    # Start TRAINING
    
    lr = utils.get_learning_rate(optimizer)
    rec, i = compute.compute(-1, model, val_loader, None, 'eval', tfh,
                                 args.backprop_batch_size, [lr, exp_name], eval_fn=my_eval_fn, debug=args.debug)

    
    for epoch in range(start_epoch, num_epochs):
        lr = utils.get_learning_rate(optimizer)
        # Pdb().set_trace()
        compute.compute(epoch, model, train_loader, optimizer, 'train', tfh,
                        args.backprop_batch_size, [lr, exp_name], eval_fn=my_eval_fn, debug=args.debug)
        rec, i = compute.compute(epoch, model, val_loader, None, 'eval', tfh,
                                 args.backprop_batch_size, [lr, exp_name], eval_fn=my_eval_fn, debug=args.debug)

        is_best = False
        utils.log('best score: {}, this score: {}'.format(best_score, rec[i]))
        # Early stopping
        if rec[i] > best_score:
            best_score = rec[i]
            is_best = True
        #

        utils.log('input to scheduler : {}'.format(1.0-1.0*rec[i]))
        my_lr_scheduler.step(1.0-1.0*rec[i], epoch=epoch)

        utils.save_checkpoint({
            'epoch': epoch,
            'best_score': best_score,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'is_best': is_best
        }, epoch, is_best, checkpoint_file, best_checkpoint_file)

    #
        if (my_lr_scheduler.shouldStopTraining()):
            print("Stop training as no improvement in accuracy - no of unconstrainedBadEopchs: {0} > {1}".format(
                my_lr_scheduler.unconstrainedBadEpochs, my_lr_scheduler.maxPatienceToStopTraining))
            break

    tfh.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='exp name',
                        type=str, default='unary')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--sample_folder', help='Sample size', type=str)
    parser.add_argument(
        '--debug', help='just load args and dont run main', action='store_true')
    parser.add_argument('--config', help='yaml config file',
                        type=str, default='../config/conll03.yml')
    parser.add_argument('--cuda', help='if cuda available, use it or not?',
                        action='store_true', default='true')

    parser.add_argument('--model', type=str, default='crf')
    parser.add_argument('--pretrained_wts', type=str, default='')

    parser.add_argument('--loss', type=str, default='nll')
    # parser.add_argument('--vocab_path',help='path to vocabulary', type=str, default = "../data/conll2003/vocab1.pkl")

    args = parser.parse_args()
    config = {}
    if os.path.exists(os.path.expanduser(args.config)):
        config = yaml.load(open(os.path.expanduser(args.config)))
    
    config.update(vars(args))
    args = utils.Map(config)
    settings.set_settings(args)
    if not args.debug:
        # pass
        main(args)






