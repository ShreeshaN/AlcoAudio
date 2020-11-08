# -*- coding: utf-8 -*-
"""
@created on: 02/16/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import argparse
import json

import torch

from alcoaudio.utils.class_utils import AttributeDict


def parse():
    parser = argparse.ArgumentParser(description="alcoaudio_configs")
    parser.add_argument('--train_net', type=bool)
    parser.add_argument('--test_net', type=bool)
    parser.add_argument('--configs_file', type=str)
    parser.add_argument('--network', type=str, choices=['convnet', 'lstm', 'crnn', 'ocnn', 'cae', 'sincnet','resnet'])
    args = parser.parse_args()
    return args


def run(args):
    if args.network == 'convnet':
        from alcoaudio.runners.convnet_runner import ConvNetRunner
        network = ConvNetRunner(args=args)
    elif args.network == 'lstm':
        from alcoaudio.runners.rnn_runner import RNNRunner
        network = RNNRunner(args=args)
    elif args.network == 'crnn':
        from alcoaudio.runners.crnn_runner import CRNNRunner
        network = CRNNRunner(args=args)
    elif args.network == 'ocnn':
        from alcoaudio.runners.ocnn_runner import OCNNRunner
        network = OCNNRunner(args=args)
    elif args.network == 'cae':
        from alcoaudio.runners.convautoencoder_runner import ConvAutoEncoderRunner
        network = ConvAutoEncoderRunner(args=args)
    elif args.network == 'sincnet':
        from alcoaudio.runners.sincnet_runner import SincNetRunner
        network = SincNetRunner(args=args)
    elif args.network == 'resnet':
        from alcoaudio.runners.resnet_pretrain import ResNetRunner
        network = ResNetRunner(args=args)
    if args.network is None:  # Default network
        from alcoaudio.runners.convnet_runner import ConvNetRunner
        network = ConvNetRunner(args=args)

    if args.train_net:
        network.train()

    if args.test_net:
        network.infer()


if __name__ == '__main__':

    args = parse().__dict__
    configs = json.load(open(args['configs_file']))
    if configs['use_gpu']:
        if not torch.cuda.is_available():
            print('GPU is not available. The program is exiting . . . ')
            exit()
        else:
            print('GPU device found: ', torch.cuda.get_device_name(0))
    configs = {**configs, **args}
    configs = AttributeDict(configs)
    run(configs)
