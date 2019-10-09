#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import scipy.stats as ss
import scipy
import numpy as np
import os
from pathlib import Path
import uproot
import re
import click

# --------------------------------------------------------------------------
# it is faster to access to root array like this dont know exactly why
def tget(t, array_name):
    return t.arrays([array_name])[array_name]

# --------------------------------------------------------------------------
def get_stat_value(s, v):
    g = r''+v+'\w+'
    a = re.search(g, s)
    if a == None:
        return -1
    a = a.group(0)[len(v):]
    return float(a)


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

    print('Filename', filename)
    f = uproot.open(filename)
    #print("List of keys: \n", f.keys())

    n_events = 1
    start_simulation_time = 0
    stop_simulation_time = 240
    try:
        stat_filename = os.path.join(Path(filename).parent, 'stat.txt')
        print('Open stat file', stat_filename)
        fs = open(stat_filename, 'r').read()
        n_events = get_stat_value(fs, '# NumberOfEvents = ')
        start_simulation_time = get_stat_value(fs, '# StartSimulationTime        = ')
        stop_simulation_time = get_stat_value(fs, '# StopSimulationTime         = ')
    except:
        print('nope')
        

    singles = f[b'Singles']
    print('nb of singles ', len(singles))

    coinc = f[b'Coincidences']
    print('nb of coincidences', len(coinc))

    delays = f[b'delay']
    print('nb of delays', len(delays))

    # plot
    fig, ax = plt.subplots(3, 3, figsize=(15, 10))

    #
    print("Detector positions by run")
    runID = tget(coinc, b'runID')
    gpx1 = tget(coinc, b'globalPosX1')
    gpx2 = tget(coinc, b'globalPosX2')
    gpy1 = tget(coinc, b'globalPosY1')
    gpy2 = tget(coinc, b'globalPosY2')
    mask = (runID == 0)
    n = 1000 # restrict to the n first values
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
    a.set_xlabel('mm')
    a.set_ylabel('mm')
    a.set_title('Transaxial detection position ({} first events only)'.format(n))

    # Axial Detection
    print('Axial Detection')
    ad1 = tget(coinc, b'globalPosZ1')
    ad2 = tget(coinc, b'globalPosZ2')
    ad = np.concatenate((ad1, ad2))
    a = ax[(0,1)]
    a.hist(ad, histtype='step', bins=100)
    a.set_xlabel('mm')
    a.set_ylabel('counts')
    a.set_title('Axial coincidences detection position')

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
    print("\tscattered", len(tsc))
    print("\tunscattered", len(tuc))
    a = ax[0,2]
    a.hist(tuc, bins=100)
    a.set_xlabel('mm')
    a.set_ylabel('counts')
    a.set_title('Axial Sensitivity Detection')
    a = ax[1,0]
    countsa, binsa = np.histogram(tsc, bins=100)
    countsr, binsr = np.histogram(z, bins=100)
    a.hist(binsa[:-1], bins=100, weights=countsa/countsr)
    a.set_xlabel('mm')
    a.set_ylabel('%')
    a.set_title('Axial Scatter fraction')

    # Delays and Randoms
    print("Delays and Randoms")
    time = tget(coinc, b'time1')
    sourceID1 = tget(coinc, b'sourceID1')
    sourceID2 = tget(coinc, b'sourceID2')
    mask = (sourceID1==0) & (sourceID2==0)
    decayF18 = time[mask]
    mask = (sourceID1==1) & (sourceID2==1)
    decayO15 = time[mask]

    ## FIXME -> measured and expected HL
    # F18 109.771(20) minutes 6586.2 sec
    # O15 122.24 seconds

    # histogram of decayO15
    bin_heights, bin_borders = np.histogram(np.array(decayO15), bins='auto', density=True)
    bin_widths = np.diff(bin_borders)
    bin_centers = bin_borders[:-1] + bin_widths / 2

    # expo fit
    def exponenial_func(x, a, b):
        return a*np.exp(-b*x)
    popt, pcov = scipy.optimize.curve_fit(exponenial_func, bin_centers, bin_heights)
    xx = np.linspace(0, 240, 240)
    yy = exponenial_func(xx, *popt)
    hl = np.log(2)/popt[1]

    # plot
    a = ax[1,1]
    a.hist(decayO15, bins=100, label='O15 HL = 122.24 sec', histtype='stepfilled', alpha=0.5, density=True)
    a.hist(decayF18, bins=100, label='F18 HL = 6586.2 sec', histtype='stepfilled', alpha=0.5, density=True)
    a.plot(xx, yy, label='O15 fit HL = {:.2f} sec'.format(hl))
    a.legend()
    a.set_xlabel('time (s)')
    a.set_ylabel('decay')
    a.set_title('Rad decays')

    # Randoms
    eventID1 = tget(coinc, b'eventID1')
    eventID2 = tget(coinc, b'eventID2')
    randoms = time[eventID1 != eventID2]
    print(len(delays))
    t1 = tget(delays, b'time1')
    print('nb of randoms', len(randoms))
    print('nb of delays', len(delays))
    a = ax[1,2]
    a.hist(randoms, bins=100, histtype='stepfilled', alpha=0.6, label='Random = {}'.format(len(randoms)))
    a.hist(t1, bins=100, histtype='step', label="Delays with coinc sorter = {}".format(len(delays)))
    a.legend()
    a.set_xlabel('time (s)')
    a.set_ylabel('events')
    a.set_title('Randoms')

    # info
    ntrue = len(tuc)
    absolute_sensitivity = ntrue/n_events
    line1 = 'Number of events {:.0f}'.format(n_events)
    line1 = line1+'\nNumber of singles {:.0f}'.format(len(singles))
    line1 = line1+'\nNumber of coincidences {:.0f}'.format(len(coinc))
    line1 = line1+'\nNumber of true {:.0f}'.format(len(tuc))
    line1 = line1+'\nNumber of randoms {:.0f}'.format(len(randoms))
    line1 = line1+'\nNumber of scatter {:.0f}'.format(len(tsc))
    line1 = line1+'\nAbsolute sensibility {:.2f} %'.format(absolute_sensitivity*100.0)
    line1 = line1+'\nStart time {:.1f} s'.format(start_simulation_time)
    line1 = line1+'\nStop time {:.1f} s'.format(stop_simulation_time)
    a = ax[2,0]
    a.plot([0], [0], '')
    a.plot([1], [1], '')
    a.set_xticks([])
    a.set_yticks([])
    a.axis('off')
    a.text(0.2, 0.5, line1)

    # end
    fig.delaxes(ax[2][1])
    fig.delaxes(ax[2][2])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --------------------------------------------------------------------------
if __name__ == '__main__':
    analyse_pet()
