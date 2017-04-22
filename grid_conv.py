# grid_conv.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import multiplot_data as mpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['axes.titlesize'] = 8

outpath = "output/revival/benchmarks17/grid_res/"

rv = 2.0
fs = 1.25

grids = [1.0, 2.0, 4.0]
grids2 = [0.0, 1.0, 2.0, 4.0]


f_mean_qu_lateral = [.20488, .2035, .19653]
print "mean lat" , f_mean_qu_lateral
# f_mean_qu_lateral[0] = 4.226 - 4.226*3.0/32.0
# f_mean_qu_lateral[1] = 4.289756 - 4.289756*1.5/16.0
# f_mean_qu_lateral[2] = 4.500 - 4.500*.75/8.0

f_exact_lateral = (4.0/3.0)*f_mean_qu_lateral[0] - (1.0/3.0)*f_mean_qu_lateral[1]

# f_max_qu_lateral = [4.32929, 4.33622, 4.52638]
# f_max_psi_lateral = [0.26082, 0.25619, 0.250869]

# f_mean_qu_conv = [0.34796, 0.322429, 0.318152]
# f_mean_qu_conv = [0.28836, 0.284357, 0.28149]
# # f_mean_qu_conv[0] = f_mean_qu_conv[0] + f_mean_qu_conv[0]*(6.25/206.25)
# # f_mean_qu_conv[1] = f_mean_qu_conv[1] + f_mean_qu_conv[1]*(12.5/212.5)
# # f_mean_qu_conv[2] = f_mean_qu_conv[2] + f_mean_qu_conv[2]*(25.0/225.0)
# # f_mean_qu_conv[0] = f_mean_qu_conv[0] + f_mean_qu_conv[0]*(6.25/212.5)
# # f_mean_qu_conv[1] = f_mean_qu_conv[1] + f_mean_qu_conv[1]*(12.5/225.0)
# # f_mean_qu_conv[2] = f_mean_qu_conv[2] + f_mean_qu_conv[2]*(25.0/250.0)
# f_mean_qu_conv = [.01287, .013238, .014518]
f_mean_qu_conv = [.0200, .0192, .0152]
f_exact_conv = (4.0/3.0)*f_mean_qu_conv[0] - (1.0/3.0)*f_mean_qu_conv[1]


f_cell_conv = [432.7, 468.4, 666.6]
f_exact_cell = (4.0/3.0)*f_cell_conv[0] - (1.0/3.0)*f_cell_conv[1]

#f = f_max_psi_lateral

def gci_final(f):
    
    print " "

    f1 = f[0]
    f2 = f[1]
    f3 = f[2]



    
    eps_12 = (f2-f1)/f1
    eps_23 = (f3-f2)/f2
    
    p = (np.log((f3-f2)/(f2-f1)))/np.log(rv)
    print "(f3-f2)/(f2-f1) =" , (f3-f2)/(f2-f1)
    #p = 1.8
    
    gci_12 = (fs*np.abs(eps_12))/((rv**p)-1.0)
    #gci_23 = (fs*np.abs(eps_23)*(rv**p))/((rv**p)-1.0)
    gci_23 = (fs*np.abs(eps_23))/((rv**p)-1.0)
    
    print "p =" , p
    # print "eps_12 =" , eps_12 , "eps_23 =" , eps_23
    print "gci_12 =" , gci_12 , "gci_23 =" , gci_23

    final = gci_23/((rv**p)*gci_12)
    return final
    

f_mean_qu_lateral_final = gci_final(f_mean_qu_lateral)
print "mean psi lateral"
print f_mean_qu_lateral_final


f_mean_qu_conv_final = gci_final(f_mean_qu_conv)
print "mean psi conv"
print f_mean_qu_conv_final

f_cell_conv_final = gci_final(f_cell_conv)
print "mean cel conv"
print f_cell_conv_final


xlim1 = -1.0
xlim2 = 5.0


# grids_labels = ['x0', 'xo/2', 'x0/4']
# grids2_labels = ['x0', 'xo/2', '', 'x0/4']
grids_labels = ['h/4', 'h/2', 'h']
grids2_labels = ['0', '0.5', '1.0', '2.0']

fig=plt.figure()



ylim1 = np.min(f_mean_qu_lateral) - 0.2*(np.max(f_mean_qu_lateral) - np.min(f_mean_qu_lateral))
ylim2 = np.max(f_mean_qu_lateral) + 0.2*(np.max(f_mean_qu_lateral) - np.min(f_mean_qu_lateral))

# ylim1 = f_exact_lateral - np.abs(0.2*(np.max(f_mean_qu_lateral) - f_exact_lateral))
# ylim2 = np.max(f_mean_qu_lateral) + np.abs(0.2*(np.max(f_mean_qu_lateral) - f_exact_lateral))

asp = np.abs((xlim2-xlim1)/(ylim2-ylim1))

ax1=fig.add_subplot(1,2,1,aspect=asp)
plt.gca().xaxis.grid(True)
plt.plot(grids, f_mean_qu_lateral,'ro-',mec='None',ms=10,zorder=10)
plt.plot([0,1], [f_exact_lateral,f_mean_qu_lateral[0]],'r-',zorder=10)
plt.plot([0], [f_exact_lateral],'r^',mec='None',ms=10,zorder=10)
plt.title('lat')
plt.xlim([xlim1,xlim2])
plt.ylim([ylim1,ylim2])
plt.xticks(grids2,grids2_labels)

ylim1 = np.min(f_mean_qu_conv) - 0.2*(np.max(f_mean_qu_conv) - np.min(f_mean_qu_conv))
ylim2 = np.max(f_mean_qu_conv) + 0.2*(np.max(f_mean_qu_conv) - np.min(f_mean_qu_conv))

# ylim1 = f_exact_conv - 0.2*(np.min(f_mean_qu_conv) - f_exact_conv)
# ylim2 = np.min(f_mean_qu_conv) + 0.2*(np.min(f_mean_qu_conv) - f_exact_conv)

asp = np.abs((xlim2-xlim1)/(ylim2-ylim1))


ax1=fig.add_subplot(2,2,2,aspect=asp)
plt.gca().xaxis.grid(True)
plt.plot(grids, f_mean_qu_conv,'bo-',mec='None',ms=10,zorder=10)
plt.plot([0,1], [f_exact_conv,f_mean_qu_conv[0]],'b-',zorder=10)
plt.plot([0], [f_exact_conv],'b^',mec='None',ms=10,zorder=10)
plt.title('conv')
plt.xlim([xlim1,xlim2])
plt.ylim([ylim2,ylim1])
plt.xticks(grids2,grids2_labels)



ylim1 = np.min(f_cell_conv) - 0.2*(np.max(f_cell_conv) - np.min(f_cell_conv))
ylim2 = np.max(f_cell_conv) + 0.2*(np.max(f_cell_conv) - np.min(f_cell_conv))

asp = np.abs((xlim2-xlim1)/(ylim2-ylim1))

ax1=fig.add_subplot(2,2,4,aspect=asp)
plt.gca().xaxis.grid(True)
plt.plot(grids, f_cell_conv,'bo-',mec='None',ms=10,zorder=10)
plt.plot([0,1], [f_exact_cell,f_cell_conv[0]],'b-',zorder=10)
plt.plot([0], [f_exact_cell],'b^',mec='None',ms=10,zorder=10)
plt.title('conv')
plt.xlim([xlim1,xlim2])
plt.ylim([ylim1,ylim2])
plt.xticks(grids2,grids2_labels)


plt.savefig(outpath+'grid_conv.eps')