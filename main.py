import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
from runners import *


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--runner', type=str, default='AnnealRunner', help='The runner to execute')
    parser.add_argument('--config', type=str, default='anneal.yml',  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='0', help='A string for documentation purpose')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-o', '--image_folder', type=str, default='images', help="The directory of image outputs")
    # NEW: pipeline mode (keeps --test working too)
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        choices=['train', 'test', 'sample', 'inpaint'],
        help='Pipeline mode. If omitted: uses --test to select train/test.'
    )

    # NEW: sampling controls (used when --mode sample)
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of images to sample')
    parser.add_argument('--sample_batch_size', type=int, default=64, help='Batch size for sampling')
    parser.add_argument('--n_steps_each', type=int, default=100, help='Langevin steps per sigma level')
    parser.add_argument('--step_lr', type=float, default=2e-5, help='Base step size for annealed Langevin dynamics')


    args = parser.parse_args()
    
    # If --mode not provided, fall back to old behavior via --test
    if args.mode is None:
        args.mode = 'test' if args.test else 'train'
    
    
    run_id = str(os.getpid())
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # args.doc = '_'.join([args.doc, run_id, run_time])
    args.log = os.path.join(args.run, 'logs', args.doc)

    # parse config file
    is_eval = args.mode in ["test", "sample", "inpaint"]
    if not is_eval:
        with open(os.path.join('configs', args.config), 'r') as f:
            config = yaml.safe_load(f)
        new_config = dict2namespace(config)
    else:
        with open(os.path.join(args.log, 'config.yml'), 'r') as f:
            config = yaml.unsafe_load(f)
        new_config = config

    if not is_eval:
        if not args.resume_training:
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            os.makedirs(args.log)

        with open(os.path.join(args.log, 'config.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    logging.info("Writing log file to {}".format(args.log))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    print(config)
    print("<" * 80)

    try:
        runner = eval(args.runner)(args, config)

        if args.mode == 'train':
            runner.train()
        elif args.mode == 'test':
            runner.test()
        elif args.mode == 'sample':
            runner.sample_pipeline()     # NEW
        elif args.mode == 'inpaint':
            runner.test_inpainting()
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
