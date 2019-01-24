#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import click
from matplotlib import pyplot as plt
import SimpleITK as sitk
import re

# -----------------------------------------------------------------------------
def read_image(folder):
    filename = os.path.join(folder, 'projection.mhd')
    img = sitk.ReadImage(filename)
    data = sitk.GetArrayFromImage(img)
    print('Read image', filename, img.GetSize(), img.GetSpacing())
    return img, data


# -----------------------------------------------------------------------------
def read_nb_events(folder):
    filename = os.path.join(folder, 'stats.txt')
    f = open(filename, 'r')
    s = f.read()
    n = '# NumberOfEvents = '
    regex = r"# NumberOfEvents = \d*"
    match = re.search(regex, s)
    n = int(s[match.start()+len(n):match.end()])
    print('Nb of events', n)
    return n


# -----------------------------------------------------------------------------
def get_psf(img, data):
    psf = np.sum(data[2, :, :], axis=1)
    #ddd = np.sum(data[imin:imax, imin:imax, :], axis=(1,2))
    x = np.arange(0,len(psf))*img.GetSpacing()[1]
    return psf, x

# -----------------------------------------------------------------------------
def get_sensitivity(data, n):
    s = np.sum(data[2, :, :])
    print('Nb of counts ', s)
    return s/n

# -----------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('folders', nargs=-1)
def plot_dose(folders):
    '''
    \b
    TODO

    \b
    <FILENAME> : input depth profile mhd

    '''

    f, ax = plt.subplots(1, 2, figsize=(10,3))

    i = 0
    for f in folders:
        img, data = read_image(f)
        nb = read_nb_events(f)
        sensitivity = get_sensitivity(data, nb)
        print(sensitivity)
        psf,x = get_psf(img, data)

        # plot psf
        a = ax[0]#[0]
        a.plot(x, psf, label=f)
        a.legend()

        # plot sensitivity
        a = ax[1]#[0]
        a.bar(0, sensitivity, width=0.2, alpha=0.3, label=f)
        a.bar(-0.5, 0)
        a.bar(0.5, 0)
        a.legend()

        i = i+1

    plt.show()



# --------------------------------------------------------------------------
if __name__ == '__main__':
    plot_dose()

