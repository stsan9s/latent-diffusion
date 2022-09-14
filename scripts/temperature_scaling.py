
import os
import torch
import argparse
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from omegaconf import OmegaConf
from ldm.data.ffhq import FFHQTestLabels
from ldm.util import instantiate_from_config
from ldm.models.diffusion.classifier import NoisyLatentImageClassifier
from ldm.modules.reliability_plot import reliability_diagram, reliability_diagrams
from pytorch_lightning.trainer import Trainer

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier_config",
        type=str,
        nargs="?",
        help="path to trained config file for classifier",
        default='none'
    )
    return parser

def get_classifier_base_dir(classifier_config):
        # remove part of path that includes the config file name and sub dir
        back_len = sum([len(name) for name in classifier_config.split('/')[-2:]])
        basedir = classifier_config[:-(back_len+1)]
        return basedir

def get_classifier_args_from_config(classifier_config):
        classifier_settings = OmegaConf.load(classifier_config)

        classifier_args = {}
        keys = ['diffusion_path', 'num_classes', 'ckpt_path', 'diffusion_ckpt_path', 'label_smoothing']
        types = [str, int, str, str, float] # matching type conversions for key args
        for k, typecast in zip(keys, types):
            if k in classifier_settings.model.params.keys():
                classifier_args[k] = typecast(classifier_settings.model.params[k])
            elif k == 'ckpt_path':
                classifier_dir = get_classifier_base_dir(classifier_config)
                # assume path to model checkpoint relative to config file
                classifier_ckpt_path = os.path.join(classifier_dir, 'checkpoints', 'last.ckpt')
                assert os.path.exists(classifier_ckpt_path), 'path to classifier checkpoint does not exist'
                classifier_args[k] = classifier_ckpt_path
            elif k == 'label_smoothing':
                classifier_args[k] = 0.0
            else:
                raise Exception(f'Missing key {k} in specified config file')

        return classifier_args

if __name__ == "__main__":

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    classifier_args = get_classifier_args_from_config(opt.classifier_config)
    classifier = NoisyLatentImageClassifier(diffusion_path=classifier_args['diffusion_path'],
                                             num_classes=classifier_args['num_classes'],
                                             ckpt_path=classifier_args['ckpt_path'],
                                             diffusion_ckpt_path=classifier_args['diffusion_ckpt_path'],
                                             label_smoothing=classifier_args['label_smoothing'],
                                             temperature_scaling=True)
    classifier.cuda()
    classifier.eval()

    trainer = Trainer(gpus=1, max_epochs=1)

    # data
    config = OmegaConf.load(opt.classifier_config)
    data = instantiate_from_config(config.data)  #  add check for test set
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # work around to train on validation for temp
    data.datasets['train'] = data.datasets['validation']

    trainer.fit(classifier, data)

    print(f'TEMPERATURE = {classifier.temperature}')
    # need better way of combining temperature into model
    torch.save(classifier.state_dict(), classifier_args['ckpt_path'] + 'tempscale') # pretty sure useless line
    torch.save(classifier.temperature, classifier_args['ckpt_path'] + '_temperature_val')
