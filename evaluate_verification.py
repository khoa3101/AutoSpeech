from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import logging
import time
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from models.model import Network
from models import resnet
from config import cfg, update_config
from utils import create_logger, Genotype
from data_objects.ZaloTestset import ZaloTestset
from utils import compute_eer
from utils import AverageMeter, ProgressMeter, accuracy

plt.switch_backend('agg')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train autospeech network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--load_path',
                        help="The path to resumed dir",
                        default=None)
    parser.add_argument('--text_arch',
                        help="The path to arch",
                        default=None)

    args = parser.parse_args()

    return args


def validate_verification(cfg, model, test_loader, out_name):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(test_loader), batch_time, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    model.eval()
    distances = []

    with torch.no_grad():
        end = time.time()
        for i, (input1, input2, path1, path2) in enumerate(test_loader):
            input1 = input1.cuda(non_blocking=True).squeeze(0)
            input2 = input2.cuda(non_blocking=True).squeeze(0)

            # compute output
            outputs1 = model(input1).mean(dim=0).unsqueeze(0)
            outputs2 = model(input2).mean(dim=0).unsqueeze(0)

            dists = F.cosine_similarity(outputs1, outputs2)
            dists = dists.data.cpu().numpy()
            distances.append('%s,%s,%f' % (path1, path2, dists))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 2000 == 0:
                progress.print(i)

    with open('%s.csv' % (out_name), 'w') as f:
        f.write('audio_1,audio_2,label\n')
        for idx, line in enumerate(distances):
            f.write(line)
            if not idx == len(distances)-1:
                f.write('\n')


def main():
    args = parse_args()
    update_config(cfg, args)
    if args.load_path is None:
        raise AttributeError("Please specify load path.")

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Set the random seed manually for reproducibility.
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

    # model and optimizer
    if cfg.MODEL.NAME == 'model':
        if args.load_path and os.path.exists(args.load_path):
            checkpoint = torch.load(args.load_path)
            genotype = checkpoint['genotype']
        else:
            raise AssertionError('Please specify the model to evaluate')
        model = Network(cfg.MODEL.INIT_CHANNELS, cfg.MODEL.NUM_CLASSES, cfg.MODEL.LAYERS, genotype)
        model.drop_path_prob = 0.0
    else:
        model = eval('resnet.{}(num_classes={})'.format(cfg.MODEL.NAME, cfg.MODEL.NUM_CLASSES))
    model = model.cuda()

    # resume && make log dir and logger
    if args.load_path and os.path.exists(args.load_path):
        checkpoint = torch.load(args.load_path)

        # load checkpoint
        model.load_state_dict(checkpoint['state_dict'])
        args.path_helper = checkpoint['path_helper']

        logger = create_logger(os.path.dirname(args.load_path))
        logger.info("=> loaded checkpoint '{}'".format(args.load_path))
    else:
        raise AssertionError('Please specify the model to evaluate')
    logger.info(args)
    logger.info(cfg)

    # dataloader
    test_dataset_verification = ZaloTestset(
        Path(cfg.DATASET.DATA_DIR), cfg.DATASET.PARTIAL_N_FRAMES
    )
    test_loader_verification = torch.utils.data.DataLoader(
        dataset=test_dataset_verification,
        batch_size=1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    validate_verification(
        cfg, model, test_loader_verification,
        args.cfg.split('/')[-1].split('.')[0]
    )



if __name__ == '__main__':
    main()
