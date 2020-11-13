directory = ""
if __name__ == '__main__':
    from os import sys, path

    directory = path.dirname(path.abspath(__file__))
    sys.path.append(path.dirname(directory))

import os
from os import listdir
from os.path import join
import argparse

import numpy
import torch

from data import Reader
from utils import *
from halo_loss_pgm import build_halo_mask
from architecture import EUnet
from trainer import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(data, model, min_points):
    """
    Estimates the 
    :param data: 
    :param model: 
    :param min_points: 
    :return: 
    """
    assert isinstance(data, Reader)

    res = []
    model.eval()

    for i in range(len(data)):
        x, y = data[i]
        x = rgba2rgb()(x, True)
        x = normalize(0.5, 0.5, )(x, True)
        x = x.transpose(2, 0, 1)[:, :240, :240]

        vx = torch.from_numpy(numpy.expand_dims(x, 0)).to(device)
        p = model(vx)
        p_numpy = p.detach().cpu().numpy()[0]

        ground_truth = get_as_list(y[:240, :240])
        instances = postprocess(p_numpy, min_points)
        detected_masks = get_as_list(instances)
        res.append(symmetric_best_dice(ground_truth, detected_masks))

    return numpy.array(res).mean(axis=0), numpy.array(res).std(axis=0)


class ReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training DeepColoring on CVPPP dataset.')
    # Path to the folder

    parser.add_argument("-s",
                        dest="basepath",
                        action=ReadableDir,
                        help="Path to CVPPP A1 dataset folder",
                        default="/media/hpc-4_Raid/vkulikov/CVPPP2017_LSC_training/training/A1/", 
                        required=True)

    args = parser.parse_args()

    basepath = args.basepath

    rgb = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_rgb.png')])
    labels = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_label.png')])


    if 0 == len(rgb):
        print("No cvppp dataset found in:" + basepath)
        exit(-1)

    # Check the names are paired correctly
    assert numpy.array([img[:-7] == lbl[:-9] for img, lbl in zip(rgb, labels)]).all() == True

    numpy.random.seed(1203412412)
    indexes = numpy.random.permutation(len(rgb))
    perm_rgb = numpy.array(rgb)[indexes].tolist()
    perm_labels = numpy.array(labels)[indexes].tolist()

    train_data = Reader(perm_rgb[:-10], perm_labels[:-10], 2)
    valid_data = Reader(perm_rgb[-10:], perm_labels[-10:], 2)

    transforms = [rgba2rgb(),
                  clip_patch((192, 192)),
                  flip_horizontally(),
                  flip_vertically(),
                  rotate90(),
                  random_transform(0.1, 90, 0),
                  blur(),
                  normalize(0.5, 0.5)]

    generator = train_data.create_batch_generator(30, transforms=transforms)
    mask_builder = build_halo_mask()

    net = EUnet(3, 9, 4, 3, 1, depth=3, padding=1, init_xavier=True, use_bn=False, use_dropout=True).to(device)
    model, errors = train(generator=generator,model=net,mask_builder=mask_builder, niter=10000,caption=join(directory, "model"))

    print(evaluate(valid_data, net, 65))
