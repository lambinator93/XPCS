# xpcs.py

### Libraries ###
import sys
import numpy as np
import h5py
import hdf5plugin
import time
import os
from scipy.optimize import curve_fit
from scipy.special import erfinv
from scipy.special import erf
from scipy import constants as sc
import scipy.io
import ipywidgets as widgets
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json



import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

from PIL import Image, ImageSequence


def asymmetric_ticks(ax, major_len=12, minor_len=6, major_w=1.9, minor_w=1.6, labelsize=22):
    # Base axis: bottom & left OUT
    ax.minorticks_on()
    ax.tick_params(which='major', length=major_len, width=major_w, labelsize=labelsize,
                   direction='out', bottom=True, left=True, top=False, right=False)
    ax.tick_params(which='minor', length=minor_len, width=minor_w,
                   direction='out', bottom=True, left=True, top=False, right=False)

    # TOP axis (IN)
    axt = ax.twiny()
    axt.set_xscale(ax.get_xscale())             # keep same scale (log/linear)
    axt.set_xlim(ax.get_xlim())
    axt.minorticks_on()
    axt.tick_params(which='major', length=major_len, width=major_w,
                    direction='in', top=True, bottom=False, labeltop=False)
    axt.tick_params(which='minor', length=minor_len, width=minor_w,
                    direction='in', top=True, bottom=False)
    for s in ('left','right','bottom'):         # show only the top spine
        axt.spines[s].set_visible(False)

    # RIGHT axis (IN)
    axr = ax.twinx()
    axr.set_yscale(ax.get_yscale())
    axr.set_ylim(ax.get_ylim())
    axr.minorticks_on()
    axr.tick_params(which='major', length=major_len, width=major_w,
                    direction='in', right=True, left=False, labelright=False)
    axr.tick_params(which='minor', length=minor_len, width=minor_w,
                    direction='in', right=True, left=False)
    for s in ('left','top','bottom'):           # show only the right spine
        axr.spines[s].set_visible(False)

    return axt, axr



def plot_g2s(ax,x,y,fit,label,marker,color,markersize=8):

    ax.semilogx(x,y,marker,markersize=markersize,color=color,label = label)
    ax.semilogx(x,fit,'-',color=color)

    # --- sizes ---
    ax.tick_params(which='major', length=7, width=1.6, labelsize=18)
    ax.tick_params(which='minor', length=4, width=1.2)

    # --- directions per side ---
    # x: top IN, bottom OUT (two calls so we don't overwrite the other side)
    ax.tick_params(axis='x', which='both',direction='in', top=True)
    ax.tick_params(axis='x', which='both',direction='out', bottom=True)

    # y: right IN, left OUT
    ax.tick_params(axis='y', which='both',direction='out', right=True)
    ax.tick_params(axis='y', which='both',direction='out', left=True)

    leg = ax.legend(frameon=False,          # remove box
            handletextpad=0.3,      # space between marker and text
            handlelength=1.0,       # length of legend handle
            labelspacing=0.2,       # vertical spacing between entries
            borderpad=0.2,          # padding inside the legend area
            fontsize=22,
            loc='lower left')       # or wherever you want



    ax.set_ylabel(r'$|F(\boldsymbol{q},t)|^2$', fontsize=28)
    ax.set_xlabel('t (s)',fontsize=28)

    # ... your plotting loop adds lines/points to ax ...
    # after plotting (and after set_xscale('log') etc.), call once:
    axt, axr = asymmetric_ticks(ax)