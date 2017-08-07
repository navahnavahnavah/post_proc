# veinplot.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import streamplot as sp
import multiplot_data as mpd
import heapq
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rcParams['axes.titlesize'] = 9
#plt.rcParams['hatch.linewidth'] = 0.1
import matplotlib.mlab as mlab


outpath = "../output/revival/summer_coarse_grid/"


# len_cm = [None, None, 3, 3, None, 3, None, None, None, None, None, 8, None,
# None, 4.5, None, None, None, None, 3, 6, 6, 6, 8, 7, 12, 5, 13, 10, 3, 9, 10,
# 12, 11, 8, 10, 8, 20, 20, 12, 12, 12, 8, 33, 33, 33, 33, 33, 9, 10, 8, 8, 8,
# 8, 30, 30, 30, 10, 14, 14, 18, 18, 18, 18, 25, 25, 25, 25, 7, 7, 7, 5, 21, 21,
# 16, 16, 16, 16, 7, 7, 4, 9, 9, 9, 10, 10, 10, 10, 13, 13, 13, 4, 22, 22, 22, 8,
# 8, 8, 8, 35, 35, 15, 19, 19, 19, 36, 36, 36, 11, 13, 30, 12, 12, 12, 22, 8, 8,
# 8, 12, 6, 5, 8, 12, 12, 12, 6, 6, 25, 25, 25, 25, 25, 25, 10, 21, 21, 21, 17, 17,
# 8, 14, 14, 14, 14, 12, 19, 18, 7, None, None, None, None, None, None, None, 8, 7,
# 6, None, None, 5, 3, 3, 5, 4, 2, 6, 4, 7, None, None, 1, 5, 2.5, 3.2, 2, None,
# None, None, None, 26, 20, 67, 67, 67, 70, 39, 4, 46, 94, 94, 94, 20, 32, 23, 50,
# 50, 21, 9, None, None, None, None, None, None, None, None, None, None, None, None,
# None, None, None, None, None, None, None, None, None, None, 22, 31, 16, 20, 20,
# 20, 9, 9, 9, 13, 15, 9, 6, 14, 38, 8, 5, 5, 26, 14, 13, 11, 13, 7, 7, 7, 7, 6, 6,
# 5, 8, 9, 3, None, None, None, None, None, None, None, None, 6, 6, 8, 13, 13, 12,
# 12, 10, 10, 10, 6, None, None, None, 4, 10, 6, 10, 8, 8, 6, None, None, None, 3,
# None, None, None, 11, 5, 7, 7, 6, 6, 6, 16, 16, 16, 10, 10, 10, 6, 6, 6, 6, 8, 8,
# 8, 8, 4, 14, 14, 10, 10, 10, 8, 9, 9, 8, 8, 8, 8, 10, 10, 9, 9, 4, 4, 4, 15, 15,
# 15, 14, 14, 14, 14, 10, 10, 10, 10, 10, 10, 7, 7, 7, 7, 10, 10, 10, 5]

len_cm = [3, 3, 3, 8,4.5, 3, 6, 6, 6, 8, 7, 12, 5, 13, 10, 3, 9, 10,
12, 11, 8, 10, 8, 20, 20, 12, 12, 12, 8, 33, 33, 33, 33, 33, 9, 10, 8, 8, 8,
8, 30, 30, 30, 10, 14, 14, 18, 18, 18, 18, 25, 25, 25, 25, 7, 7, 7, 5, 21, 21,
16, 16, 16, 16, 7, 7, 4, 9, 9, 9, 10, 10, 10, 10, 13, 13, 13, 4, 22, 22, 22, 8,
8, 8, 8, 35, 35, 15, 19, 19, 19, 36, 36, 36, 11, 13, 30, 12, 12, 12, 22, 8, 8,
8, 12, 6, 5, 8, 12, 12, 12, 6, 6, 25, 25, 25, 25, 25, 25, 10, 21, 21, 21, 17, 17,
8, 14, 14, 14, 14, 12, 19, 18, 7, 8, 7,6, 5, 3, 3, 5, 4, 2, 6, 4, 7, 1, 5, 2.5, 3.2, 2,
26, 20, 67, 67, 67, 70, 39, 4, 46, 94, 94, 94, 20, 32, 23, 50, 50, 21, 9,
22, 31, 16, 20, 20, 20, 9, 9, 9, 13, 15, 9, 6, 14, 38, 8, 5, 5, 26, 14, 13,
11, 13, 7, 7, 7, 7, 6, 6, 5, 8, 9, 3, 6, 6, 8, 13, 13, 12,
12, 10, 10, 10, 6, 4, 10, 6, 10, 8, 8, 6, 3, 11, 5, 7, 7, 6, 6, 6, 16, 16, 16,
10, 10, 10, 6, 6, 6, 6, 8, 8, 8, 8, 4, 14, 14, 10, 10, 10, 8, 9, 9,
8, 8, 8, 8, 10, 10, 9, 9, 4, 4, 4, 15, 15, 15, 14, 14, 14, 14, 10, 10, 10, 10,
10, 10, 7, 7, 7, 7, 10, 10, 10, 5]

dip_angle = np.array([85, 90, 46, 40, 40, 30, 60, 60, 64, 80, 70, 43, 30, 42, 19, 70,
85, 74, 74, 60, 50, 47, 80, 70, 6, 67, 90, 18, 30, 41, 5, 70, 10, 50, 30, 55,
40, 52, 45, 60, 85, 45, 40, 60, 68, 70, 25, 65, 35, 70, 20, 78, 18, 15, 15, 21,
77, 85, 20, 0, 61, 66, 90, 33, 80, 5, 5, 21, 19, 34, 85, 63, 75, 70, 85, 5, 5,
85, 35, 30, 90, 50, 84, 60, 90, 70, 75, 90, 68, 20, 60, 35, 60, 70, 53, 90, 75,
5, 70, 15, 20, 55, 45, 60, 60, 45, 48, 30, 80, 90, 0, 65, 30, 65, 90, 85, 40, 90,
90, 45, 45, 10, 90, 0, 0, 90, 80, 0, 16, 0, 90, 40, 35, 80, 0, 70, 65, 90, 90,
60, 60, 50, 90, 70, 90, 90, 75, 65, 40, 35, 90, 35, 70, 60])

fig=plt.figure(figsize=(14.0,8.0))

# vein/fracture histogram
ax=fig.add_subplot(3, 2, 1, frameon=True)

n, bins, patches = plt.hist(len_cm, 25, facecolor='green', alpha=0.75)

# diffusion time = f(len) plot

lens = np.arange(0.0, 100.0, 1.0)
d_a = 2.5e-13 # m^2/s
t_diffs_a = np.zeros(len(lens))
for i in range(len(lens)):
    t_diffs_a[i] = ((lens[i]/100.0) * (lens[i]/100.0)) / ((3.14e7)*d_a)

ax=fig.add_subplot(3, 2, 3, frameon=True)
ax.grid(True)
plt.plot(lens, np.log10(t_diffs_a), lw=3, c='g')
# plt.xlabel('vein/fracture length [cm]',fontsize=9)
plt.ylabel('log10 of diffusion time',fontsize=9)
ax.set_xticklabels([])
plt.yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ['1 yr', '10 yrs', '100 yrs', '1000 yrs', '10000 yrs', '100000 yrs'])



ax=fig.add_subplot(3, 2, 5, frameon=True)
ax.grid(True)
plt.plot(lens, t_diffs_a/1000.0, lw=3, c='g')
plt.xlabel('vein/fracture length [cm]',fontsize=9)
plt.ylabel('diffusion time [kyr]',fontsize=9)
# plt.yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], ['1 yr', '10 yrs', '100 yrs', '1000 yrs', '10000 yrs'])



# r = np.arange(0, 2, 0.01)
#theta = np.zeros(len(dip_angle))
#theta = 2 * np.pi * dip_angle*2.0/360.0

theta = dip_angle * np.pi * 2.0 / 360.0
print theta
the_bins = np.arange(0.0,90.0,5.0)
print " "
print "the bins" , the_bins
the_bins = 2.0*np.pi*the_bins/360.0
hist, bin_edges = np.histogram(theta, bins=the_bins)

ax = plt.subplot(1, 2, 2, projection='polar')
ax.grid(True)
ax.scatter(the_bins, np.ones(len(the_bins)), s=3.0*(hist**1.5), facecolor='g', edgecolor='k')
ax.set_rmax(np.pi/2)
ax.set_yticks([])

plt.subplots_adjust( wspace=0.2, hspace=0.2, bottom=0.1, top=0.9, left=0.1, right=0.9)
plt.savefig(outpath+"vein_plot.png")
