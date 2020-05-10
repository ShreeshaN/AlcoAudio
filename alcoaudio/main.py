# -*- coding: utf-8 -*-
"""
@created on: 02/16/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

from alcoaudio.utils.class_utils import AttributeDict
import json
import argparse


def parse():
    parser = argparse.ArgumentParser(description="alcoaudio_configs")
    parser.add_argument('--train_net', type=bool)
    parser.add_argument('--test_net', type=bool)
    parser.add_argument('--configs_file', type=str)
    parser.add_argument('--network', type=str, choices=['convnet', 'lstm', 'crnn', 'ocnn', 'cae'])
    args = parser.parse_args()
    return args


def run(args):
    if network == 'convnet':
        from alcoaudio.runners.convnet_runner import ConvNetRunner
        network = ConvNetRunner(args=args)
    elif network == 'lstm':
        from alcoaudio.runners.convnet_runner import ConvNetRunner
        network = ConvNetRunner(args=args)
    elif network == 'crnn':
        from alcoaudio.runners.convnet_runner import ConvNetRunner
        network = ConvNetRunner(args=args)
    elif network == 'ocnn':
        from alcoaudio.runners.convnet_runner import ConvNetRunner
        network = ConvNetRunner(args=args)
    elif network == 'cae':
        from alcoaudio.runners.convnet_runner import ConvNetRunner
        network = ConvNetRunner(args=args)
    if network is None:
        from alcoaudio.runners.convnet_runner import ConvNetRunner
        network = ConvNetRunner(args=args)

    if args.train_net:
        network.train()

    if args.test_net:
        network.test()


if __name__ == '__main__':
    args = parse().__dict__
    configs = json.load(open(args['configs_file']))
    configs = {**configs, **args}
    configs = AttributeDict(configs)
    run(configs)
