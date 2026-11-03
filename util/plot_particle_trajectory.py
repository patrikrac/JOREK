#!/usr/bin/env python
"""
Plot particle trajectory from part*.h5.
"""
from __future__ import print_function

import sys
import numpy as np
from glob import glob

import h5py
import matplotlib
matplotlib.use("Agg") # Headless backend
import matplotlib.pyplot as plt


def plot_trajectory(group_name, particle_nums=np.s_[:]):

    files = glob('part*.h5')
    with h5py.File(files[0], 'r') as f:
        num_particles = f[group_name+'x'][particle_nums].shape[0]

    out = np.empty((3,len(files),num_particles), dtype=np.float32)

    for i, file in enumerate(sorted(files)):
        with h5py.File(file, 'r') as f:
            out[:,i,:] = f[group_name+'x'][particle_nums].T

    # Plot overlapping t with transparency for a measure of the distribution
    plt.plot(out[0,:,:], out[1,:,:], lw=1, alpha=0.1)
    plt.xlabel("$R$")
    plt.ylabel("$Z$")
    plt.savefig('particle_traces.png')
    print('Wrote particle_traces.png')
    plt.clf()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sel = list(map(int, sys.argv[1:]))
    else:
        sel = np.s_[:]
    plot_trajectory('/groups/001/', sel)
