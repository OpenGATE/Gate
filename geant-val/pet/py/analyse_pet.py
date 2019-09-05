#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import os
from pathlib import Path
import uproot
import click

# --------------------------------------------------------------------------
# it is faster to access to root array like this dont know exactly why
def tget(t, array_name):
    return t.arrays([array_name])[array_name]

# --------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('filename', nargs=1)
def analyse_pet(filename):
    '''
    \b
    PET analysis:
    
    <FILENAME> : input root filename
    '''

    print(filename)
    f = uproot.open(filename)
    #print("List of keys: \n", f.keys())

    singles = f[b'Singles']
    print('nb singles ', len(singles))
    
    coinc = f[b'Coincidences']
    print('nb coincidences', len(coinc))

    # plot 
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    #
    print("Detector positions by run")
    runID = tget(coinc, b'runID')
    gpx1 = tget(coinc, b'globalPosX1')
    gpx2 = tget(coinc, b'globalPosX2')
    gpy1 = tget(coinc, b'globalPosY1')
    gpy2 = tget(coinc, b'globalPosY2')
    print('run', len(runID))
    mask = (runID == 0)
    n = 2000 # restrict to the n first values 
    r0_gpx1 = gpx1[mask][:n]
    r0_gpx2 = gpx2[mask][:n]
    r0_gpy1 = gpy1[mask][:n]
    r0_gpy2 = gpy2[mask][:n]
    r0x = np.concatenate((r0_gpx1,r0_gpx2, r0_gpx1))
    r0y = np.concatenate((r0_gpy1,r0_gpy2, r0_gpy1))
    a = ax[(0,0)]
    a.scatter(r0x, r0y, s=1)
    mask = (runID == 1)
    r1_gpx1 = gpx1[mask][:n]
    r1_gpx2 = gpx2[mask][:n]
    r1_gpy1 = gpy1[mask][:n]
    r1_gpy2 = gpy2[mask][:n]
    r1x = np.concatenate((r1_gpx1,r1_gpx2, r1_gpx1))
    r1y = np.concatenate((r1_gpy1,r1_gpy2, r1_gpy1))
    a = ax[(0,0)]
    a.scatter(r1x, r1y, s=1)
    a.set_aspect('equal', adjustable='box')

    
    # Axial Detection
    print('Axial Detection')
    ad1 = tget(coinc, b'globalPosZ1')
    ad2 = tget(coinc, b'globalPosZ2')
    ad = np.concatenate((ad1, ad2))
    a = ax[(0,1)]
    a.hist(ad, bins=100)
    a.set_title('Axial detection position')

    # True unscattered coincidences (tuc)
    # True scattered coincindences (tsc)
    print('True scattered and unscattered coincindences')
    z = (ad1+ad2)/2
    compt1 = tget(coinc, b'comptonPhantom1')
    compt2 = tget(coinc, b'comptonPhantom2')
    rayl1 = tget(coinc, b'RayleighPhantom1')
    rayl2 = tget(coinc, b'RayleighPhantom2')
    mask =  ((compt1==0) & (compt2==0) & (rayl1==0) & (rayl2==0))
    tuc = z[mask]
    tsc = z[~mask]
    print("scattered", len(tsc))
    print("unscattered", len(tuc))
    a = ax[0,2]
    a.hist(tuc, bins=100)
    a.set_title('Axial Sensitivity Detection')
    a = ax[1,0]
    a.hist(tsc, bins=100)
    a.set_title('Axial Scatter Detection')

    # Delays and Randoms
    print("Delays and Randoms")
    delays = tget(coinc, b'time1')
    eventID1 = tget(coinc, b'eventID1')
    eventID2 = tget(coinc, b'eventID2')
    randoms = delays[eventID1 != eventID2]
    print('delays', len(delays))
    print('randoms', len(randoms))
    a = ax[1,1]
    a.hist(delays, bins=100)
    a.set_title('Delays')
    a = ax[1,2]
    a.hist(randoms, bins=100)
    a.set_title('Randoms')

    # decay ?

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    
# --------------------------------------------------------------------------
if __name__ == '__main__':
    analyse_pet()

