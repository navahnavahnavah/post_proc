# flow_history.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 11

plot_col = ['#000000', '#940000', '#d26618', '#dfa524', '#9ac116', '#139a59', '#35b5aa', '#0740d2', '#7f05d4', '#b100de']

outpath = "../output/revival/winter_basalt_box/"

nsteps = 100
n_hs = 3
n_dx = 3
n_hist = 10
n_hist_active = 2

param_c = 4200.0
param_lambda = 1.2
param_hs = 5.0 # sedimentation rate 5m/Myr
param_hb = np.zeros(n_hist) # aquifer thickness 500m
param_rho = 1000.0
param_dx = 50000.0

steps = np.arange(nsteps)
age_vec = np.linspace(0.0, 5.0, nsteps)

hs_vec = np.zeros([len(steps),n_hist])
dx_vec = np.zeros([len(steps),n_hist])

q_vec = np.zeros([len(steps),n_hist])
q_vec_log10 = np.zeros([len(steps),n_hist])
q_vec_log10_myr = np.zeros([len(steps),n_hist])


txt_labels = ['sed 5m/Myr, 500m basement', 'sed 50m/Myr, 500m basement', 'sed 5m/Myr, 100m basement', 'sed 50m/Myr, 100m basement', 'sed 100m fixed, 500m basement', 'sed 200m fixed, 500m basement', 'sed 50m/yr, 500m basement, dx fixed at 10km', 'sed 50m/yr, 500m basement, dx fixed at 20km', 'sed 50m/yr, 500m basement, dx fixed at 100km', 'sed 100m/yr, 500m basement, dx fixed at 100km']


# generate qm/ql (hfs_vec)
hfs_vec = np.zeros([len(steps),n_hist])
for i in range(nsteps):
    # n_hist = 0 : fisher2000, sediment 5m/Myr, basement 500m
    hfs_vec[i,0] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,0] = 5.0 * age_vec[i]
    dx_vec[i,0] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[0] = 500.0

    # n_hist = 1 : fisher2000, sediment 50m/Myr, basement 500m
    hfs_vec[i,1] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,1] = 50.0 * age_vec[i]
    dx_vec[i,1] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[1] = 500.0

    # n_hist = 2 : fisher2000, sediment 5m/Myr, basement 100m
    hfs_vec[i,2] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,2] = 5.0 * age_vec[i]
    dx_vec[i,2] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[2] = 100.0

    # n_hist = 3 : fisher2000, sediment 50m/Myr, basement 100m
    hfs_vec[i,3] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,3] = 50.0 * age_vec[i]
    dx_vec[i,3] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[3] = 100.0

    # n_hist = 4 : sediment fixed at 100m, basement 500
    hfs_vec[i,4] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,4] = 100.0
    dx_vec[i,4] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[4] = 500.0

    # n_hist = 5 : sediment fixed at 200m, basement 500
    hfs_vec[i,5] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,5] = 200.0
    dx_vec[i,5] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[5] = 500.0

    # n_hist = 6 : sediment 50m/Myr, dx fixed at 10km
    hfs_vec[i,6] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,6] = 50.0 * age_vec[i]
    dx_vec[i,6] = 10000.0 #(5000.0 + 500.0*age_vec[i])/2.0
    param_hb[6] = 500.0

    # n_hist = 7 : sediment 50m/Myr, dx fixed at 20km
    hfs_vec[i,7] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,7] = 50.0 * age_vec[i]
    dx_vec[i,7] = 20000.0 #(5000.0 + 500.0*age_vec[i])/2.0
    param_hb[7] = 500.0

    # n_hist = 8 : sediment 50m/Myr, dx fixed at 100km
    hfs_vec[i,8] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,8] = 50.0 * age_vec[i]
    dx_vec[i,8] = 100000.0 #(5000.0 + 500.0*age_vec[i])/2.0
    param_hb[8] = 500.0

    # n_hist = 9 : sediment 100m/Myr, dx fixed at 100km
    hfs_vec[i,9] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,9] = 100.0 * age_vec[i]
    dx_vec[i,9] = 100000.0 #(5000.0 + 500.0*age_vec[i])/2.0
    param_hb[9] = 500.0





print "age_vec shape" , age_vec.shape
print "hfs_vec shape" , hfs_vec.shape


for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for i in range(len(steps)):
        q_vec[i,j] = (-1.0 * param_lambda * dx_vec[i,j] / (hs_vec[i,j] * param_hb[j] * param_rho * param_c)) * np.log(-hfs_vec[i,j] + 1.0)
        q_vec_log10[i,j] = np.log10(q_vec[i,j])
        q_vec_log10_myr[i,j] = np.log10(q_vec[i,j]*(3.14e7))
    print "done with" , j

# print "q_vec" , q_vec
# print "q_vec_log10 4" , q_vec_log10[:,4]

the_lw = 2.0
m_size = 50
fig=plt.figure(figsize=(13.0,13.0))

ax1=fig.add_subplot(2, 2, 1, frameon=True)


plt.plot(age_vec,q_vec_log10[:,0], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::10],q_vec_log10[::10,0], marker='o', zorder=5, facecolor='k', s=m_size, label=txt_labels[0])

plt.plot(age_vec,q_vec_log10[:,1], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::10],q_vec_log10[::10,1], marker='s', zorder=5, facecolor='k', s=m_size, label=txt_labels[1])

plt.plot(age_vec,q_vec_log10[:,2], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::10],q_vec_log10[::10,2], marker='o', zorder=5, facecolor='none', s=m_size, label=txt_labels[2])

plt.plot(age_vec,q_vec_log10[:,3], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::10],q_vec_log10[::10,3], marker='s', zorder=5, facecolor='none', s=m_size, label=txt_labels[3])

plt.plot(age_vec,q_vec_log10[:,4], color=plot_col[1], label=txt_labels[4], lw=the_lw)

plt.plot(age_vec,q_vec_log10[:,5], color=plot_col[2], label=txt_labels[5], lw=the_lw)

plt.plot(age_vec,q_vec_log10[:,6], color=plot_col[3], label=txt_labels[6], lw=the_lw)

plt.plot(age_vec,q_vec_log10[:,7], color=plot_col[4], label=txt_labels[7], lw=the_lw)

plt.plot(age_vec,q_vec_log10[:,8], color=plot_col[5], label=txt_labels[8], lw=the_lw)

plt.plot(age_vec,q_vec_log10[:,9], color=plot_col[6], label=txt_labels[9], lw=the_lw)

#ax1.tick_params('y', colors='b')
plt.ylabel('log10(q [m/s])', color='b')
# plt.legend(fontsize=8)
plt.legend(fontsize=8,bbox_to_anchor=(1.6, 1.0),ncol=1,columnspacing=0.1)

plt.xlim([np.min(age_vec),np.max(age_vec)])

# ax2 = ax1.twinx()
# ax2.plot(age_vec,q_vec_log10_myr[:,0], 'k', lw=2.25)
# #plt.scatter(age_vec[::10,0],q_vec_log10_myr[::10,0], marker='o', s=40, facecolor='m', edgecolor='none', zorder=5)
# ax2.tick_params('y', colors='r')
# ax2.set_ylabel('q [m/yr]', color='r')
# #plt.yticks([0.0, 1.0, 2.0], [1, 10, 100])





#plt.plot()

ax1.set_xlabel('age [Myr]')


plt.savefig(outpath+"q_history.png",bbox_inches='tight')
