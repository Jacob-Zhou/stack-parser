# -*- coding: utf-8 -*-

import argparse
import os
from parser.cmds import Evaluate, Predict, Train
from parser.config import Config

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train()
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument('--conf', '-c', default='config.ini',
                               help='path to config file')
        subparser.add_argument('--file', '-f', default='exp/conll09',
                               help='path to saved files')
        subparser.add_argument('--model', '-m', default='model',
                               help='model filename')
        subparser.add_argument('--vocab', '-v', default='vocab',
                               help='vocab filename')
        subparser.add_argument('--device', '-d', default='-1',
                               help='ID of GPU to use')
        subparser.add_argument('--preprocess', '-p', action='store_true',
                               help='whether to preprocess the data only')
        subparser.add_argument('--seed', '-s', default=1, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads', '-t', default=4, type=int,
                               help='max num of threads')
    args = parser.parse_args()

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    print(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Override the default configs with parsed arguments")
    args.vocab = os.path.join(args.file, args.vocab)
    args.model = os.path.join(args.file, args.model)
    config = Config(args.conf)
    config.update(vars(args))
    print(config)

    print(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(config)
