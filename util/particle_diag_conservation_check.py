#!/usr/bin/env python
"""
Plot particle timetraces from diag.h5, useful to check conservation laws.
"""
from __future__ import print_function

import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg") # Headless backend
import matplotlib.pyplot as plt


def plot_diag_conservation(f, group_name, basename):
    dset = group_name + basename
    # Plot histogram of min/max ratio
    if False:
        plt.hist(f[dset].min(axis=1)/f[dset].max(axis=1), bins=20, alpha=0.75)
        plt.xlabel('%s min/max'%basename)
        plt.ylabel('fraction')
        plt.grid(True)
        plt.savefig('%s_minmax.png'%basename)
        print("Wrote %s_minmax.png"%basename)
        plt.clf()

    # Plot histogram of growth rate
    if False:
        plt.hist((f[dset][:, -1] - f[dset][:, 0])/f[dset][:, 0], bins=20, alpha=0.75)
        plt.xlabel('%s growth rate'%basename)
        plt.ylabel('fraction')
        plt.grid(True)
        plt.savefig('%s_growth.png'%basename)
        print("Wrote %s_growth.png"%basename)
        plt.clf()

    # Plot histogram of variance
    if False:
        #plt.hist(np.std(ma.std(t,axis=1), bins=20, alpha=0.75)
        plt.xlabel('%s stddev'%basename)
        plt.ylabel('fraction')
        plt.grid(True)
        plt.savefig('%s_std.png'%basename)
        print("Wrote %s_std.png"%basename)
        plt.clf()

    # Plot overlapping t with transparency for a measure of the distribution
    relative_change = f[dset]/f[dset][0, :] - 1 # relative change
    plt.plot(f[group_name+'t'], relative_change, lw=1, alpha=0.1)
    plt.xlabel("$t$")
    plt.ylabel("%s"%basename.replace('_', '\_'));
    plt.savefig('%s_trace.png'%basename)
    print("Wrote %s_trace.png"%basename)
    # Calculate maximum relative difference
    print('Max delta: %s'%np.amax(np.abs(relative_change)))
    print('mean abs: %s'%np.mean(np.abs(relative_change)))
    plt.clf()


if __name__ == "__main__":
    # Read the type from cli arguments
    with h5py.File('diag.h5', 'r') as f:
        for VAR_NAME in sys.argv[1:]:
            plot_diag_conservation(f, '/groups/001/', VAR_NAME)
