# plot_temp_ends.py


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import streamplot as sp
import multiplot_data as mpd
import heapq
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams['axes.titlesize'] = 12

# plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

plot_col = ['#940000', '#d26618', '#dfa524', '#9ac116', '#139a31', '#35b5aa', '#0740d2', '#7f05d4', '#b100de']

#poop: path
outpath = "../output/revival/local_fp_output/"
x = np.linspace(0.0,100000.0,2001)
age_hist = np.linspace(0.25,4.25,len(x),endpoint=True)

#hack: CALC: Q_lith paths
q_lith = np.zeros([len(x),10])
q_lith_age = np.zeros([len(x),10])
starts = np.linspace(0.25,0.75,11,endpoint=True)

for i in range(len(x)):
    for j in range(10):
        q_lith[i,j] = 0.5 / (((starts[j]+0.035*(x[1]-x[0])*i/1000.0)**0.5))
        q_lith_age[i,j] = 0.5 / (age_hist[i]**0.5)

#todo: FIG: plot Q_lith
fig=plt.figure(figsize=(12.0,12.0))

ax=fig.add_subplot(2, 2, 1, frameon=True)
for j in range(6):
    plt.plot(x, q_lith[:,j], c=plot_col[j], label='start with '+str(starts[j]), lw=1.5)
plt.xlabel('distance from axis [m]')
plt.ylabel('lithospheric heat flux [W/m2]')

ax=fig.add_subplot(2, 2, 2, frameon=True)
for j in range(1):
    plt.plot(age_hist, q_lith_age[:,j], c=plot_col[j], label='start with '+str(starts[j]), lw=1.5)
plt.xlabel('crust age [Myr]')
plt.ylabel('lithospheric heat flux [W/m2]')

plt.legend(bbox_to_anchor=(1.0,1.0),fontsize=8)
plt.savefig(outpath+'z_Q_lith.png',bbox_inches='tight')



#hack: CALC: cond_hist
sed_hist = np.zeros([len(x),10])

sed_hist[:,0] = 100.0 * age_hist
sed_hist[:,2] = 50.0 * age_hist


cond_t_top = np.zeros([len(x),10])
cond_t_bottom = np.zeros([len(x),10])
cond_t_mean = np.zeros([len(x),10])

for i in range(len(x)):
    cond_t_top[i,0] = ((q_lith_age[i,0] * (sed_hist[i,0]))/(1.2)) + 2.0
    cond_t_bottom[i,0] = ((q_lith_age[i,0] * (sed_hist[i,0]+200.0))/(1.2)) + 2.0
    cond_t_mean[i,0] = (cond_t_top[i,0] + cond_t_bottom[i,0])/2.0

    q_fixed_temp = np.mean(q_lith_age[:,0])
    cond_t_top[i,1] = ((q_fixed_temp * (sed_hist[i,0]))/(1.2)) + 2.0
    cond_t_bottom[i,1] = ((q_fixed_temp * (sed_hist[i,0]+200.0))/(1.2)) + 2.0
    cond_t_mean[i,1] = (cond_t_top[i,1] + cond_t_bottom[i,1])/2.0

    cond_t_top[i,2] = ((q_lith_age[i,0] * (sed_hist[i,2]))/(1.2)) + 2.0
    cond_t_bottom[i,2] = ((q_lith_age[i,0] * (sed_hist[i,2]+200.0))/(1.2)) + 2.0
    cond_t_mean[i,2] = (cond_t_top[i,2] + cond_t_bottom[i,2])/2.0

    q_ratio = 0.4
    cond_t_top[i,3] = ((q_ratio*q_lith_age[i,0] * (sed_hist[i,0]))/(1.2)) + 2.0
    cond_t_bottom[i,3] = ((q_ratio*q_lith_age[i,0] * (sed_hist[i,0]+200.0))/(1.2)) + 2.0
    cond_t_mean[i,3] = (cond_t_top[i,3] + cond_t_bottom[i,3])/2.0

    cond_t_top[i,4] = ((q_ratio*q_lith_age[i,0] * (sed_hist[i,2]))/(1.2)) + 2.0
    cond_t_bottom[i,4] = ((q_ratio*q_lith_age[i,0] * (sed_hist[i,2]+200.0))/(1.2)) + 2.0
    cond_t_mean[i,4] = (cond_t_top[i,4] + cond_t_bottom[i,4])/2.0

    q_ratio = 0.6
    cond_t_top[i,5] = ((q_ratio*q_lith_age[i,0] * (sed_hist[i,0]))/(1.2)) + 2.0
    cond_t_bottom[i,5] = ((q_ratio*q_lith_age[i,0] * (sed_hist[i,0]+200.0))/(1.2)) + 2.0
    cond_t_mean[i,5] = (cond_t_top[i,5] + cond_t_bottom[i,5])/2.0

    cond_t_top[i,6] = ((q_ratio*q_lith_age[i,0] * (sed_hist[i,2]))/(1.2)) + 2.0
    cond_t_bottom[i,6] = ((q_ratio*q_lith_age[i,0] * (sed_hist[i,2]+200.0))/(1.2)) + 2.0
    cond_t_mean[i,6] = (cond_t_top[i,6] + cond_t_bottom[i,6])/2.0

#todo: FIG: plot conduction temps
fig=plt.figure(figsize=(16.0,12.0))

ax=fig.add_subplot(2, 2, 1, frameon=True)

plt.plot(age_hist, cond_t_top[:,0], c='b', label='cond_t_top', lw=1.5)
plt.plot(age_hist, cond_t_mean[:,0], c='#d69a00', label='cond_t_mean', lw=1.5)
plt.plot(age_hist, cond_t_bottom[:,0], c='r', label='cond_t_bottom', lw=1.5)

plt.plot(age_hist, cond_t_top[:,1], c='b', lw=1.5, linestyle='--')
plt.plot(age_hist, cond_t_mean[:,1], c='#d69a00',lw=1.5, linestyle='--')
plt.plot(age_hist, cond_t_bottom[:,1], c='r',lw=1.5, linestyle='--')

plt.plot(age_hist, cond_t_top[:,2], c='b', lw=4.5, linestyle=':')
plt.plot(age_hist, cond_t_mean[:,2], c='#d69a00',lw=4.5, linestyle=':')
plt.plot(age_hist, cond_t_bottom[:,2], c='r',lw=4.5, linestyle=':')

plt.plot(age_hist, cond_t_top[:,3], c='b', lw=1.5, linestyle='-.')
plt.plot(age_hist, cond_t_mean[:,3], c='#d69a00',lw=1.5, linestyle='-.')
plt.plot(age_hist, cond_t_bottom[:,3], c='r',lw=1.5, linestyle='-.')

plt.xlabel('crust age [Myr]')
plt.ylabel('temperature in flow layer [C]')

plt.legend(bbox_to_anchor=(1.0,1.0),fontsize=8)


ax=fig.add_subplot(2, 2, 2, frameon=True)

# plt.plot(age_hist, cond_t_top[:,0], c='b', label='cond_t_top', lw=1.5)
# plt.plot(age_hist, cond_t_mean[:,0], c='#d69a00', label='cond_t_mean', lw=1.5)
# plt.plot(age_hist, cond_t_bottom[:,0], c='r', label='cond_t_bottom', lw=1.5)
plt.fill_between(age_hist, cond_t_top[:,0], cond_t_bottom[:,0], facecolor='#88cefb', lw=0, zorder=0, alpha=0.35, label='cooling lithosphere + fast sedimentation')

#
# plt.plot(age_hist, cond_t_top[:,1], c='b', lw=1.5, linestyle='--')
# plt.plot(age_hist, cond_t_mean[:,1], c='#d69a00',lw=1.5, linestyle='--')
# plt.plot(age_hist, cond_t_bottom[:,1], c='r',lw=1.5, linestyle='--')
ax.fill_between(age_hist, cond_t_top[:,1], cond_t_bottom[:,1], facecolor='#fa5555', alpha=0.35, lw=0, zorder=0, label='fixed lithospheric heat + fast sedimentation')

# plt.plot(age_hist, cond_t_top[:,2], c='b', lw=4.5, linestyle=':')
# plt.plot(age_hist, cond_t_mean[:,2], c='#d69a00',lw=4.5, linestyle=':')
# plt.plot(age_hist, cond_t_bottom[:,2], c='r',lw=4.5, linestyle=':')
ax.fill_between(age_hist, cond_t_top[:,2], cond_t_bottom[:,2], facecolor='#9aeb26', alpha=0.35, lw=0, zorder=0, label='cooling lithosphere + slow sedimentation')

# plt.plot(age_hist, cond_t_top[:,3], c='b', lw=1.5, linestyle='-')
plt.plot(age_hist, cond_t_mean[:,3], c='k',lw=1.5, linestyle='-', zorder=5)
# plt.plot(age_hist, cond_t_bottom[:,3], c='r',lw=1.5, linestyle='-')
#ax.fill_between(age_hist, cond_t_top[:,3], cond_t_bottom[:,3], facecolor='#808080', alpha=0.45, lw=0, zorder=0, label='cooling lithosphere, fast sed, lateral heat diverted')
ax.fill_between(age_hist, cond_t_top[:,3], cond_t_bottom[:,3], facecolor='none', hatch='///', alpha=1.0, lw=0, zorder=2, label='cooling lithosphere, fast sed, lateral heat diverted')


# plt.plot(age_hist, cond_t_top[:,4], c='b', lw=1.5, linestyle='--')
plt.plot(age_hist, cond_t_mean[:,4], c='k',lw=1.5, linestyle='--', zorder=5)
# plt.plot(age_hist, cond_t_bottom[:,4], c='r',lw=1.5, linestyle='--')
ax.fill_between(age_hist, cond_t_top[:,4], cond_t_bottom[:,4], facecolor='#737373', alpha=0.45, lw=0, zorder=1, label='cooling lithosphere, slow sed, lateral heat diverted')
#ax.fill_between(age_hist, cond_t_top[:,4], cond_t_bottom[:,4], facecolor='none', hatch='o', alpha=1.0, lw=0, zorder=0, label='cooling lithosphere, slow sed, lateral heat diverted')

plt.xlim([0.25,4.25])
plt.ylim([0.0,200.0])

plt.xlabel('crust age [Myr]')
plt.ylabel('temperature in flow layer [C]')





ax3=fig.add_subplot(2, 2, 3, frameon=True)


aa = plt.plot(age_hist, cond_t_mean[:,3], color='b',lw=1.5, linestyle='--', zorder=5, label='0.4, cooling lithosphere, fast sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,3], cond_t_bottom[:,3], facecolor='none', edgecolor='b', hatch='//', alpha=1.0, lw=0, zorder=2, label='0.4, cooling lithosphere, fast sed, lateral heat diverted')


bb = plt.plot(age_hist, cond_t_mean[:,4], color='b',lw=1.5, linestyle='-', zorder=5, label='0.4, cooling lithosphere, slow sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,4], cond_t_bottom[:,4], facecolor='#3cbeee', alpha=0.45, lw=0, zorder=1, label='0.4, cooling lithosphere, slow sed, lateral heat diverted')

cc = plt.plot(age_hist, cond_t_mean[:,5], color='r',lw=1.5, linestyle='--', zorder=5, label='0.6, cooling lithosphere, fast sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,5], cond_t_bottom[:,5], facecolor='none', hatch='\\', alpha=1.0, lw=0, edgecolor='r', zorder=2, label=' 0.6, cooling lithosphere, fast sed, lateral heat diverted')


dd = plt.plot(age_hist, cond_t_mean[:,6], color='r',lw=1.5, linestyle='-', zorder=5, label='0.6, cooling lithosphere, slow sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,6], cond_t_bottom[:,6], facecolor='#d50f0f', alpha=0.3, lw=0, zorder=1, label='0.6, cooling lithosphere, slow sed, lateral heat diverted')

plt.xlim([0.25,4.25])
plt.ylim([0.0,150.0])

plt.xlabel('crust age [Myr]')
plt.ylabel('temperature in flow layer [C]')

plt.legend(bbox_to_anchor=(1.0,1.0),fontsize=10)





ax3=fig.add_subplot(2, 2, 4, frameon=True)


#aa = plt.plot(age_hist, cond_t_mean[:,3], color='b',lw=1.5, linestyle='--', zorder=5, label='0.4, cooling lithosphere, fast sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,3], cond_t_mean[:,3], facecolor='none', edgecolor='b', hatch='//', alpha=1.0, lw=0, zorder=2, label='0.4, cooling lithosphere, fast sed, lateral heat diverted')


#bb = plt.plot(age_hist, cond_t_mean[:,4], color='b',lw=1.5, linestyle='-', zorder=5, label='0.4, cooling lithosphere, slow sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,4], cond_t_mean[:,4], facecolor='#3cbeee', alpha=0.45, lw=0, zorder=1, label='0.4, cooling lithosphere, slow sed, lateral heat diverted')

#cc = plt.plot(age_hist, cond_t_mean[:,5], color='r',lw=1.5, linestyle='--', zorder=5, label='0.6, cooling lithosphere, fast sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,5], cond_t_mean[:,5], facecolor='none', hatch='++', alpha=0.8, lw=0, edgecolor='r', zorder=2, label=' 0.6, cooling lithosphere, fast sed, lateral heat diverted')


#dd = plt.plot(age_hist, cond_t_mean[:,6], color='r',lw=1.5, linestyle='-', zorder=5, label='0.6, cooling lithosphere, slow sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,6], cond_t_mean[:,6], facecolor='#d50f0f', alpha=0.3, lw=0, zorder=1, label='0.6, cooling lithosphere, slow sed, lateral heat diverted')

plt.xlim([0.25,4.25])
plt.ylim([0.0,80.0])

plt.xlabel('crust age [Myr]')
plt.ylabel('temperature in flow layer [C]')

plt.legend(bbox_to_anchor=(1.0,1.0),fontsize=10)

plt.savefig(outpath+'z_T_cond.png',bbox_inches='tight')
plt.savefig(outpath+'z_T_cond.eps',bbox_inches='tight')







#todo: FIG: plot cond temps norm
fig=plt.figure(figsize=(16.0,12.0))

ax=fig.add_subplot(2, 2, 1, frameon=True)

plt.plot(age_hist, cond_t_top[:,0], c='b', label='cond_t_top', lw=1.5)
plt.plot(age_hist, cond_t_mean[:,0], c='#d69a00', label='cond_t_mean', lw=1.5)
plt.plot(age_hist, cond_t_bottom[:,0], c='r', label='cond_t_bottom', lw=1.5)

plt.plot(age_hist, cond_t_top[:,1], c='b', lw=1.5, linestyle='--')
plt.plot(age_hist, cond_t_mean[:,1], c='#d69a00',lw=1.5, linestyle='--')
plt.plot(age_hist, cond_t_bottom[:,1], c='r',lw=1.5, linestyle='--')

plt.plot(age_hist, cond_t_top[:,2], c='b', lw=4.5, linestyle=':')
plt.plot(age_hist, cond_t_mean[:,2], c='#d69a00',lw=4.5, linestyle=':')
plt.plot(age_hist, cond_t_bottom[:,2], c='r',lw=4.5, linestyle=':')

plt.plot(age_hist, cond_t_top[:,3], c='b', lw=1.5, linestyle='-.')
plt.plot(age_hist, cond_t_mean[:,3], c='#d69a00',lw=1.5, linestyle='-.')
plt.plot(age_hist, cond_t_bottom[:,3], c='r',lw=1.5, linestyle='-.')

plt.xlabel('crust age [Myr]')
plt.ylabel('temperature in flow layer [C]')
plt.xlim([0.25,4.25])
plt.ylim([0.0,200.0])
plt.yticks(np.arange(0.0,225.0,25.0))

plt.legend(bbox_to_anchor=(1.0,1.0),fontsize=8)


ax=fig.add_subplot(2, 2, 2, frameon=True)

# plt.plot(age_hist, cond_t_top[:,0], c='b', label='cond_t_top', lw=1.5)
# plt.plot(age_hist, cond_t_mean[:,0], c='#d69a00', label='cond_t_mean', lw=1.5)
# plt.plot(age_hist, cond_t_bottom[:,0], c='r', label='cond_t_bottom', lw=1.5)
plt.fill_between(age_hist, cond_t_top[:,0], cond_t_bottom[:,0], facecolor='#88cefb', lw=0, zorder=0, alpha=0.35, label='cooling lithosphere + fast sedimentation')

#
# plt.plot(age_hist, cond_t_top[:,1], c='b', lw=1.5, linestyle='--')
# plt.plot(age_hist, cond_t_mean[:,1], c='#d69a00',lw=1.5, linestyle='--')
# plt.plot(age_hist, cond_t_bottom[:,1], c='r',lw=1.5, linestyle='--')
ax.fill_between(age_hist, cond_t_top[:,1], cond_t_bottom[:,1], facecolor='#fa5555', alpha=0.35, lw=0, zorder=0, label='fixed lithospheric heat + fast sedimentation')

# plt.plot(age_hist, cond_t_top[:,2], c='b', lw=4.5, linestyle=':')
# plt.plot(age_hist, cond_t_mean[:,2], c='#d69a00',lw=4.5, linestyle=':')
# plt.plot(age_hist, cond_t_bottom[:,2], c='r',lw=4.5, linestyle=':')
ax.fill_between(age_hist, cond_t_top[:,2], cond_t_bottom[:,2], facecolor='#9aeb26', alpha=0.35, lw=0, zorder=0, label='cooling lithosphere + slow sedimentation')

# plt.plot(age_hist, cond_t_top[:,3], c='b', lw=1.5, linestyle='-')
plt.plot(age_hist, cond_t_mean[:,3], c='k',lw=1.5, linestyle='-', zorder=5)
# plt.plot(age_hist, cond_t_bottom[:,3], c='r',lw=1.5, linestyle='-')
#ax.fill_between(age_hist, cond_t_top[:,3], cond_t_bottom[:,3], facecolor='#808080', alpha=0.45, lw=0, zorder=0, label='cooling lithosphere, fast sed, lateral heat diverted')
ax.fill_between(age_hist, cond_t_top[:,3], cond_t_bottom[:,3], facecolor='none', hatch='///', alpha=1.0, lw=0, zorder=2, label='cooling lithosphere, fast sed, lateral heat diverted')


# plt.plot(age_hist, cond_t_top[:,4], c='b', lw=1.5, linestyle='--')
plt.plot(age_hist, cond_t_mean[:,4], c='k',lw=1.5, linestyle='--', zorder=5)
# plt.plot(age_hist, cond_t_bottom[:,4], c='r',lw=1.5, linestyle='--')
ax.fill_between(age_hist, cond_t_top[:,4], cond_t_bottom[:,4], facecolor='#737373', alpha=0.45, lw=0, zorder=1, label='cooling lithosphere, slow sed, lateral heat diverted')
#ax.fill_between(age_hist, cond_t_top[:,4], cond_t_bottom[:,4], facecolor='none', hatch='o', alpha=1.0, lw=0, zorder=0, label='cooling lithosphere, slow sed, lateral heat diverted')

plt.xlim([0.25,4.25])
plt.ylim([0.0,200.0])
plt.yticks(np.arange(0.0,225.0,25.0))

plt.xlabel('crust age [Myr]')
plt.ylabel('temperature in flow layer [C]')





ax3=fig.add_subplot(2, 2, 3, frameon=True)


aa = plt.plot(age_hist, cond_t_mean[:,3], color='b',lw=1.5, linestyle='--', zorder=5, label='0.4, cooling lithosphere, fast sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,3], cond_t_bottom[:,3], facecolor='none', edgecolor='b', hatch='//', alpha=1.0, lw=0, zorder=2, label='0.4, cooling lithosphere, fast sed, lateral heat diverted')


bb = plt.plot(age_hist, cond_t_mean[:,4], color='b',lw=1.5, linestyle='-', zorder=5, label='0.4, cooling lithosphere, slow sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,4], cond_t_bottom[:,4], facecolor='#3cbeee', alpha=0.45, lw=0, zorder=1, label='0.4, cooling lithosphere, slow sed, lateral heat diverted')

cc = plt.plot(age_hist, cond_t_mean[:,5], color='r',lw=1.5, linestyle='--', zorder=5, label='0.6, cooling lithosphere, fast sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,5], cond_t_bottom[:,5], facecolor='none', hatch='\\', alpha=1.0, lw=0, edgecolor='r', zorder=2, label=' 0.6, cooling lithosphere, fast sed, lateral heat diverted')


dd = plt.plot(age_hist, cond_t_mean[:,6], color='r',lw=1.5, linestyle='-', zorder=5, label='0.6, cooling lithosphere, slow sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,6], cond_t_bottom[:,6], facecolor='#d50f0f', alpha=0.3, lw=0, zorder=1, label='0.6, cooling lithosphere, slow sed, lateral heat diverted')

plt.xlim([0.25,4.25])
plt.ylim([0.0,200.0])
plt.yticks(np.arange(0.0,225.0,25.0))

plt.xlabel('crust age [Myr]')
plt.ylabel('temperature in flow layer [C]')

plt.legend(bbox_to_anchor=(1.0,1.0),fontsize=10)





ax3=fig.add_subplot(2, 2, 4, frameon=True)


#aa = plt.plot(age_hist, cond_t_mean[:,3], color='b',lw=1.5, linestyle='--', zorder=5, label='0.4, cooling lithosphere, fast sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,3], cond_t_mean[:,3], facecolor='none', edgecolor='b', hatch='//', alpha=1.0, lw=0, zorder=2, label='0.4, cooling lithosphere, fast sed, lateral heat diverted')


#bb = plt.plot(age_hist, cond_t_mean[:,4], color='b',lw=1.5, linestyle='-', zorder=5, label='0.4, cooling lithosphere, slow sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,4], cond_t_mean[:,4], facecolor='#3cbeee', alpha=0.45, lw=0, zorder=1, label='0.4, cooling lithosphere, slow sed, lateral heat diverted')

#cc = plt.plot(age_hist, cond_t_mean[:,5], color='r',lw=1.5, linestyle='--', zorder=5, label='0.6, cooling lithosphere, fast sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,5], cond_t_mean[:,5], facecolor='none', hatch='++', alpha=0.8, lw=0, edgecolor='r', zorder=2, label=' 0.6, cooling lithosphere, fast sed, lateral heat diverted')


#dd = plt.plot(age_hist, cond_t_mean[:,6], color='r',lw=1.5, linestyle='-', zorder=5, label='0.6, cooling lithosphere, slow sed, lateral heat diverted')
ax3.fill_between(age_hist, cond_t_top[:,6], cond_t_mean[:,6], facecolor='#d50f0f', alpha=0.3, lw=0, zorder=1, label='0.6, cooling lithosphere, slow sed, lateral heat diverted')

plt.xlim([0.25,4.25])
plt.ylim([0.0,200.0])
plt.yticks(np.arange(0.0,225.0,25.0))

plt.xlabel('crust age [Myr]')
plt.ylabel('temperature in flow layer [C]')

plt.legend(bbox_to_anchor=(1.0,1.0),fontsize=10)

plt.savefig(outpath+'z_T_cond_norm.png',bbox_inches='tight')
plt.savefig(outpath+'z_T_cond_norm.eps',bbox_inches='tight')
