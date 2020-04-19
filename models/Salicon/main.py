import numpy as np
from scipy.misc import imsave
from os import listdir, makedirs
from os.path import isfile, join
import sys, getopt

from Salicon import SALICONtf


def main(img_dir,out_dir):
    img_dir = ''
    out_dir = ''


    makedirs(out_dir, exist_ok=True)

    images = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]

    s = SALICONtf('model_weights.pt')

    for img_name in images:
        smap = s.compute_saliency(img_path=join(img_dir, img_name))
        imsave(join(out_dir, img_name), smap)

def call(header,output) :
    main(header,output)
