# load classifier
# get dataloader for test set
# pick specific timestep t
# for timestep t, check classifier accuracy/confidence over test set
# plot on reliability plot - by confidence to accuracy
# get all pairs for the max softmax score as (confidence_score, win_or_lost)
    # plot on histogram based on bins of confidence score
    # for each bin calculate average accuracy and confidence, plot

import os
import torch
import argparse
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from omegaconf import OmegaConf
from collections import OrderedDict
from ldm.data.ffhq import FFHQTestLabels
from ldm.util import instantiate_from_config
from ldm.models.diffusion.classifier import NoisyLatentImageClassifier
from ldm.modules.reliability_plot import reliability_diagram, reliability_diagrams
from pytorch_lightning.trainer import Trainer

# Override matplotlib default styling.
plt.style.use("seaborn")

plt.rc("font", size=12)
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)

plt.rc("axes", titlesize=16)
plt.rc("figure", titlesize=16)

def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier_config",
        type=str,
        nargs="?",
        help="path to trained config file for classifier",
        default='none'
    )
    parser.add_argument(
        "--temperature_scaling",
        type=str2bool,
        nargs="?",
        const=True,
        help="use temp scaling",
        default=False,
    )
    return parser

def get_classifier_base_dir(classifier_config):
        # remove part of path that includes the config file name and sub dir
        back_len = sum([len(name) for name in classifier_config.split('/')[-2:]])
        basedir = classifier_config[:-(back_len+1)]
        return basedir

def get_classifier_args_from_config(classifier_config, temperature_scaling=False):
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
                if temperature_scaling:
                    classifier_ckpt_path = os.path.join(classifier_dir, 'checkpoints', 'last.ckpttempscale')
                else:
                    classifier_ckpt_path = os.path.join(classifier_dir, 'checkpoints', 'last.ckpt')
                assert os.path.exists(classifier_ckpt_path), 'path to classifier checkpoint does not exist'
                classifier_args[k] = classifier_ckpt_path
            elif k == 'label_smoothing':
                classifier_args[k] = 0.0
            else:
                raise Exception(f'Missing key {k} in specified config file')

        return classifier_args

def generate_reliability_diagram(df, plot_name):

    title = "\n".join(plot_name.split())

    y_true = df.true_labels.values
    y_pred = df.pred_labels.values
    y_conf = df.confidences.values

    fig = reliability_diagram(y_true, y_pred, y_conf, num_bins=10, draw_ece=True,
                             draw_bin_importance="alpha", draw_averages=True,
                             title=title, figsize=(6, 6), dpi=100, 
                             return_fig=True)
    return fig

if __name__ == "__main__":

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    classifier_args = get_classifier_args_from_config(opt.classifier_config, opt.temperature_scaling)
    classifier = NoisyLatentImageClassifier(diffusion_path=classifier_args['diffusion_path'],
                                            num_classes=classifier_args['num_classes'],
                                            ckpt_path=classifier_args['ckpt_path'],
                                            diffusion_ckpt_path=classifier_args['diffusion_ckpt_path'],
                                            label_smoothing=classifier_args['label_smoothing'],
                                            temperature_scaling=opt.temperature_scaling)
    print(f'TEMPERATURE IS ==== {classifier.temperature}')
    classifier.cuda()
    classifier.eval()


    trainer = Trainer(gpus=1)

    # data
    config = OmegaConf.load(opt.classifier_config)
    data = instantiate_from_config(config.data)  #  add check for test set
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    results = OrderedDict()

    for timestep in (250, 500, 750):
        classifier.set_test_timestep(timestep)
        predictions = trainer.predict(classifier, dataloaders=data)

        confidence, targets = predictions[0]
        for c, t in predictions[1:]:
            confidence = torch.vstack((confidence, c))
            targets = torch.cat((targets, t))

        max_confs, argmax_inds = confidence.max(dim=1)
        
        max_confs = max_confs.numpy()
        argmax_inds = argmax_inds.numpy()
        targets = targets.numpy()

        reliability_metrics = {'true_labels': targets,
                               'pred_labels': argmax_inds,
                               'confidences': max_confs}
        df = pd.DataFrame(reliability_metrics)
        classifier_base_dir = get_classifier_base_dir(opt.classifier_config)
        sub_dir = os.path.join(classifier_base_dir, 'confidence_calibration')
        if opt.temperature_scaling:
            sub_dir = os.path.join(classifier_base_dir, 'confidence_calibration', 'temperature')
        os.makedirs(sub_dir, exist_ok=True)
        csv_name = f'confidence_t-{timestep}.csv'
        df.to_csv(os.path.join(sub_dir, csv_name))

        # create individual reliability diagram for timestep
        plot_name = csv_name.split('.')[0]
        fig = generate_reliability_diagram(df, plot_name)
        fig.savefig(os.path.join(sub_dir, f'{plot_name}.png'), bbox_inches='tight')

        results[plot_name] = reliability_metrics


    # save plot for all timesteps in a grid
    fig = reliability_diagrams(results, num_bins=10, draw_bin_importance="alpha",
                               num_cols=3, dpi=100, return_fig=True)
    fig.savefig(os.path.join(sub_dir, 'all_timesteps.png'))
