# multi_lines.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import math
import multiplot_data as mpd
from numpy.ma import masked_array
from mpl_toolkits.axes_grid1 import AxesGrid
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
# plt.rc('title', fontsize=8)
# plt.rc('ylabel', labelsize=8)
# plt.rc('xlabel', labelsize=8)


plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"


steps = 10

path1 = "output/revival/multi_push_orhs400/"
path2 = "output/revival/multi_push_orhs400/"
outpath = "output/revival/multi_push_orhs400/"

x = np.loadtxt(path1 + 'x.txt',delimiter='\n')
y = np.loadtxt(path1 + 'y.txt',delimiter='\n')

glass1_0 = np.loadtxt(path1 + 'z_pri_glass.txt')
glass1_0 = glass1_0*110.3839/2.7

glass2_0 = np.loadtxt(path2 + 'z_pri_glass.txt')
glass2_0 = glass2_0*110.3839/2.7


mask = np.loadtxt(path1 + 'maskP.txt')

perm0 = np.loadtxt(path1 + 'permMat.txt')
perm0 = np.log10(perm0)


def cut_chem(geo0,index):
    geo_cut_chem = geo0[:,(index*len(x)):(index*len(x)+len(x))]
    return geo_cut_chem
    

    
    
    
for i in range(0,steps,1):     
    
    glass1 = cut_chem(glass1_0,i)
    glass2 = cut_chem(glass2_0,i)
    perm = perm0[:,i*len(x):((i)*len(x)+len(x))]
    
    
    
    mask_aq = np.ones(mask.shape)
    for jj in range(len(x)-1):
        for j in range(len(y)-1):
            if perm[j,jj] > -13:
                mask_aq[j,jj] = 1.0
            if perm[j,jj] < -14:
                mask_aq[j,jj] = 0.0
                
    
    glass1_x = np.zeros(len(x))
    glass2_x = np.zeros(len(x))
    
    for j in range(len(x)):
        glass1_x[j] = np.sum(glass1[:,j])/(np.max(glass1)*np.sum(mask_aq[:,j]))
        glass2_x[j] = np.sum(glass2[:,j])/(np.max(glass2)*np.sum(mask_aq[:,j]))
        
    print "glass1"
    print glass1_x
    print " "
    print "glass2"
    print glass2_x
    print " "
    print " "
        
        
    fig=plt.figure()
    
    ax1=fig.add_subplot(2,1,1)
    plt.plot(x[10:-10],glass1_x[10:-10],'r')
    plt.plot(x[10:-10],glass2_x[10:-10],'b')
    
    ax1=fig.add_subplot(2,1,2)
    plt.pcolor(mask_aq)
    plt.colorbar()
    
    plt.savefig(outpath+'ml_'+str(i)+'.png')
    
    
    
    
    