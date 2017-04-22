# box_batches.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import box_temps as bt
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=14)
plt.rcParams['axes.titlesize'] = 12
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)



col = ['r', 'darkorange', 'gold', 'lawngreen', 'g', 'c', 'b', 'purple', 'm', 'hotpink', 'gray']

secondary = np.array(['', 'stilbite', 'aragonite', 'kaolinite', 'albite', 'saponite_mg', 'celadonite',
'clinoptilolite', 'pyrite', 'mont_na', 'goethite', 'dolomite', 'smectite', 'saponite_k',
'anhydrite', 'siderite', 'calcite', 'quartz', 'kspar', 'saponite_na', 'nont_na', 'nont_mg',
'nont_k', 'nont_h', 'nont_ca', 'muscovite', 'mesolite', 'hematite', 'mont_ca', 'verm_ca',
'analcime', 'phillipsite', 'diopside', 'epidote', 'gismondine', 'hedenbergite', 'chalcedony',
'verm_mg', 'ferrihydrite', 'natrolite', 'talc', 'smectite_low', 'prehnite', 'chlorite',
'scolecite', 'chamosite7a', 'clinochlore14a', 'clinochlore7a', 'saponite_ca', 'verm_na',
'pyrrhotite', 'magnetite', 'lepidocrocite', 'daphnite_7a', 'daphnite_14a', 'verm_k',
'mont_k', 'mont_mg'])

density = np.array([0, 2.15, 2.93, 2.63, 2.62, 2.3, 3.0, 
2.15, 5.02, 2.01, 4.27, 2.84, 2.01, 
2.3, 2.97, 3.96, 2.71, 2.65, 2.56, 2.3, 
2.3, 2.3, 2.3, 2.3, 2.3, 2.81, 2.29, 5.3,
2.01, 2.5, 2.27, 2.2, 3.3, 3.41, 2.26, 
3.56, 2.65, 2.5, 3.8, 2.23, 2.75, 2.01, 
2.87, 2.468, 2.27, 3.0, 3.0, 3.0, 2.3,
2.5, 4.62, 5.15, 4.08, 3.2, 3.2, 2.5,
2.01, 2.01])

molar = np.array([0, 480.19, 100.19, 258.16, 263.02, 480.19, 429.02, 
2742.13, 119.98, 549.07, 88.85, 180.4, 540.46, 
480.19, 136.14, 115.86, 100.19, 60.08, 278.33, 480.19, 
495.9, 495.9, 495.9, 495.9, 495.9, 398.71, 1164.9, 159.69,
549.07, 504.19, 220.15, 704.93, 216.55, 519.3, 718.55, 
248.08, 60.08, 504.19, 169.7, 380.22, 379.27, 540.46, 
395.38, 67.4, 392.34, 664.18, 595.22, 595.22, 480.19,
504.19, 85.12, 231.53, 88.85, 664.18, 664.18, 504.19,
49.07, 549.07])

n_box = 3
tn = 250

age = np.linspace(0.0,5.0,2500)
age2 = np.linspace(0.0,5.0,tn)

tra_float = np.array([11.00, 11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00])
tra_str = ['11.00', '11.25', '11.50', '11.75', '12.00', '12.25', '12.50', '12.75', '13.00']
tra_str2 = ['11', '11.25', '11.5', '11.75', '12', '12.25', '12.5', '12.75', '13']

# xb_float = np.array([-1.0, -1.25, -1.50, -1.75, -2.00, -2.25, -2.50, -2.75, -3.00])
# xb_str = ['-1.0', '-1.25', '-1.50', '-1.75', '-2.00', '-2.25', '-2.50', '-2.75', '-3.00']
xb_float = np.array([-3.00, -2.75, -2.50, -2.25, -2.00, -1.75, -1.50, -1.25, -1.00])
xb_str = ['-3.00', '-2.75', '-2.50', '-2.25', '-2.00', '-1.75', '-1.50', '-1.25', '-1.00']
xb_str2 = ['1000', '-2.75', '-2.50', '-2.25', '100', '-1.75', '-1.50', '-1.25', '10']

param0_float = tra_float
param0_str = tra_str

param1_float = xb_float
param1_str = xb_str

all_mins = " "
the_count = 0

mat_output = np.zeros([len(param0_str)+1, len(param1_str)+1])
final_alt_vol_a = np.zeros([len(param0_str)+1, len(param1_str)+1])
final_alt_vol_b = np.zeros([len(param0_str)+1, len(param1_str)+1])
final_alt_vol_c = np.zeros([len(param0_str)+1, len(param1_str)+1])
final_alt_vol_s = np.zeros([len(param0_str)+1, len(param1_str)+1])

path0 = "basalt_box_output/batch2/"

secondary_mat = np.zeros([58,n_box,tn])
secondary_mat_vol = np.zeros([58,n_box,tn])
alt_vol = np.zeros(secondary_mat_vol[0,:,:].shape)
print alt_vol.shape

for i in range(len(param0_str)):
    for j in range(len(param1_str)):
        
        path_ij = 'tra' + param0_str[i] + 'xb' + param1_str[j] + '/'
        mat_output[j,i] = param0_float[i]*param1_float[j]
        print " "
        print path_ij
        #print "mat_output[j,i]" , mat_output[j,i]
        
        the_count = the_count + 1
        if the_count == 1:
            sum_mat = np.transpose(np.loadtxt(path0 + path_ij + 'medium_sum.txt'))
        
        # PRIMARY FOR EACH SIM
        primary_mat5 = np.transpose(np.loadtxt(path0 + path_ij + 'primary_mat5.txt'))*110.0/2.9
        
        # ALTERATION VOLUME FOR EACH SIM
        
        for k in range(1,58):
            if os.path.isfile(path0 + path_ij + 'secondary_mat' + str(k) + '.txt'):
                secondary_mat[k,:,:] = np.transpose(np.loadtxt(path0 + path_ij + 'secondary_mat' + str(k) + '.txt'))#*molar[i]/density[i]
                if not str(secondary[k]) in all_mins:
                    all_mins = all_mins + str(secondary[k]) + " "
        secondary_mat_sum = np.sum(secondary_mat[:,:-1,:],axis=1)
        
        for k in range(1,58):
            secondary_mat_vol[k,:,:] = secondary_mat[k,:,:]*molar[k]/density[k]
            
        alt_vol[0,:] = np.sum(secondary_mat_vol[:,0,:], axis=0)
        alt_vol[1,:] = np.sum(secondary_mat_vol[:,1,:], axis=0)
        alt_vol[2,:] = np.sum(secondary_mat_vol[:,2,:], axis=0)
        final_alt_vol_a[j,i] = alt_vol[0,-1]/(alt_vol[0,-1]+primary_mat5[0,-1])
        final_alt_vol_b[j,i] = alt_vol[1,-1]/(alt_vol[1,-1]+primary_mat5[1,-1])
        final_alt_vol_c[j,i] = (alt_vol[0,-1] + alt_vol[1,-1])/(alt_vol[0,-1]+alt_vol[1,-1]+primary_mat5[0,-1]+primary_mat5[1,-1])
        final_alt_vol_s[j,i] = alt_vol[2,-1]/(alt_vol[2,-1]+primary_mat5[2,-1])
        #print "final" , final_alt_vol_a[j,i]

        
        
### CONTOUR PLOT REFERENCE        
# asp = (np.max(param1_float) - np.min(param1_float))/(np.max(param0_float) - np.min(param0_float))
# print "asp" , asp
# ax1=fig.add_subplot(2,4,5, aspect=asp)
# the_plot = plt.contourf(param0_float, param1_float, final_alt_vol_a[:-1,:-1])
# cbar = plt.colorbar(the_plot, orientation='vertical',fraction=0.046, pad=0.04)
# plt.xticks(param0_float[::2])
# plt.yticks(param1_float[::2])
# ax1.set_xticklabels(param0_float[::2])
# ax1.set_yticklabels(param1_float[::2])
# plt.xlim([np.min(param0_float),np.max(param0_float)])
# plt.ylim([np.min(param1_float),np.max(param1_float)])   

print "heyyyy"
print all_mins

final_alt_vol_a = final_alt_vol_a*100.0
final_alt_vol_b = final_alt_vol_b*100.0

final_max=0
if np.max(final_alt_vol_a) > np.max(final_alt_vol_b):
    final_max = np.max(final_alt_vol_a)
if np.max(final_alt_vol_a) < np.max(final_alt_vol_b):
    final_max = np.max(final_alt_vol_b)

final_min = 0
if np.min(final_alt_vol_a[:-1,:-1]) <= np.min(final_alt_vol_b[:-1,:-1]):
    final_min = np.min(final_alt_vol_a[:-1,:-1])
if np.min(final_alt_vol_a[:-1,:-1]) >= np.min(final_alt_vol_b[:-1,:-1]):
    final_min = np.min(final_alt_vol_b[:-1,:-1])

final_min = 0.75
final_max = 4.0

fig=plt.figure()
w, h = plt.figaspect(0.34) 
fig.set_figheight(h) 
fig.set_figwidth(w) 
###################################
###### FINAL ALT VOL A PLOTS ######
###################################


print "final_alt_vol_a"

param0_pcol = np.append(param0_float, param0_float[-1]+(param0_float[1] - param0_float[0]))
param1_pcol = np.append(param1_float, param1_float[-1]+(param1_float[1] - param1_float[0]))


asp = (np.max(param1_pcol) - np.min(param1_pcol))/(np.max(param0_pcol) - np.min(param0_pcol))
print "asp" , asp
ax1=fig.add_subplot(1,4,2,aspect=asp)
the_plot = plt.pcolor(param0_pcol, param1_pcol, final_alt_vol_a, vmin=final_min, vmax=final_max, cmap=cm.YlGn)
the_plot.set_edgecolor('face')
#cbar = plt.colorbar(the_plot, orientation='vertical',fraction=0.046, pad=0.04)
plt.xticks(param0_float[::2] + (param0_float[1] - param0_float[0])/2.0)
plt.yticks(param1_float[::4] + (param1_float[1] - param1_float[0])/2.0)
ax1.set_xticklabels(tra_str2[::2])
ax1.set_yticklabels(xb_str2[::4])
plt.xlim([np.min(param0_pcol),np.max(param0_pcol)])
plt.ylim([np.min(param1_pcol),np.max(param1_pcol)])
plt.title('Chamber A')
plt.xlabel('log$_{10}$(t$_{RA}$ [s]), residence time in A')
plt.ylabel('t$_{RB}$/t$_{RA}$, ratio of residence times')



###################################
###### FINAL ALT VOL B PLOTS ######
###################################


print "final_alt_vol_b"

param0_pcol = np.append(param0_float, param0_float[-1]+(param0_float[1] - param0_float[0]))
param1_pcol = np.append(param1_float, param1_float[-1]+(param1_float[1] - param1_float[0]))


asp = (np.max(param1_pcol) - np.min(param1_pcol))/(np.max(param0_pcol) - np.min(param0_pcol))
print "asp" , asp
ax1=fig.add_subplot(1,4,3,aspect=asp)
the_plot = plt.pcolor(param0_pcol, param1_pcol, final_alt_vol_b, vmin=final_min, vmax=final_max, cmap=cm.YlGn)
the_plot.set_edgecolor('face')
#cbar = plt.colorbar(the_plot, orientation='vertical',fraction=0.046, pad=0.04)
plt.xticks(param0_float[::2] + (param0_float[1] - param0_float[0])/2.0)
plt.yticks(param1_float[::4] + (param1_float[1] - param1_float[0])/2.0)
ax1.set_xticklabels(tra_str2[::2])
ax1.set_yticklabels(xb_str2[::4])
plt.xlim([np.min(param0_pcol),np.max(param0_pcol)])
plt.ylim([np.min(param1_pcol),np.max(param1_pcol)])
plt.title('Chamber B')
plt.xlabel('log$_{10}$(t$_{RA}$ [s])')

cbar_ax = fig.add_axes([0.30, 0.07, 0.37, 0.04])
cbar = fig.colorbar(the_plot, cax=cbar_ax,orientation='horizontal',ticks=np.linspace(final_min,final_max,8))
plt.title('Percent Alteration Volume')
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")


#######################################
###### FINAL ALT VOL COMBO PLOTS ######
#######################################


print "final_alt_vol_c"
param0_pcol = np.append(param0_float, param0_float[-1]+(param0_float[1] - param0_float[0]))
param1_pcol = np.append(param1_float, param1_float[-1]+(param1_float[1] - param1_float[0]))


asp = (np.max(param1_pcol) - np.min(param1_pcol))/(np.max(param0_pcol) - np.min(param0_pcol))
print "asp" , asp
ax1=fig.add_subplot(1,4,4,aspect=asp)
the_plot = plt.pcolor(param0_pcol, param1_pcol, final_alt_vol_c/final_alt_vol_s, vmin=0.6, vmax=1.4, cmap=cm.bwr)
the_plot.set_edgecolor('face')
#cbar = plt.colorbar(the_plot, orientation='vertical',fraction=0.046, pad=0.04)
plt.xticks(param0_float[::2] + (param0_float[1] - param0_float[0])/2.0)
plt.yticks(param1_float[::4] + (param1_float[1] - param1_float[0])/2.0)
ax1.set_xticklabels(tra_str2[::2])
ax1.set_yticklabels(xb_str2[::4])
plt.xlim([np.min(param0_pcol),np.max(param0_pcol)])
plt.ylim([np.min(param1_pcol),np.max(param1_pcol)])
plt.title('Dual Chamber / Solo Chamber')
plt.xlabel('log$_{10}$(t$_{RA}$ [s])')

cbar_ax = fig.add_axes([0.755, 0.07, 0.21, 0.04])
enhance_ticks =  np.round(np.linspace(np.min(final_alt_vol_c[:-1,:-1]/final_alt_vol_s[:-1,:-1]),np.max(final_alt_vol_c[:-1,:-1]/final_alt_vol_s[:-1,:-1]),10),2)
print enhance_ticks
enhance_ticks = np.linspace(0.6,1.4,9)
cbar = fig.colorbar(the_plot, cax=cbar_ax,orientation='horizontal',ticks=enhance_ticks)
plt.title('Dual-chamber relative to solo-chamber alteration')
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")


####################
## TEMP EVOLUTION ##
####################

temp_ax = fig.add_axes([0.04, 0.57, 0.16, 0.3])

print age.shape
print bt.temp_100_ht.shape
temp_plot = temp_ax.plot(age,bt.temp_100_ht,color='b',lw=2,label='Mean hydrothermal activity')
temp_plot = temp_ax.plot(age,bt.temp_100_2ht,color='b',lw=1,linestyle='-',label='2 x mean hydrothermal activity')
plt.ylim([0,100.0])
#plt.xlabel('Crust Age [Ma]', fontsize=10)
plt.ylabel('Aquifer Temp. [$^o$C]', fontsize=10)
plt.yticks(fontsize=9)
#plt.legend(ncol=1,fontsize=9,loc='best')


###################
## SUM EVOLUTION ##
###################

temp_ax = fig.add_axes([0.04, 0.17, 0.16, 0.3])
sum_ticks = np.round(np.linspace(np.min(sum_mat),np.max(sum_mat),8),2)
sum_ticks = np.linspace(0.96,1.00,5)
sum_plot = temp_ax.plot(age2,sum_mat[0,:],color='magenta',lw=2,label='Chamber A')
sum_plot = temp_ax.plot(age2,sum_mat[1,:],color='darkorange',lw=2,label='Chamber B')
sum_plot = temp_ax.plot(age2,sum_mat[2,:],color='c',lw=2,label='Solo chamber')
plt.xlabel('Crust Age [Ma]', fontsize=10)
plt.ylabel('Normalized basalt surface area', fontsize=10)
plt.yticks(sum_ticks,fontsize=9)
plt.xticks(fontsize=9)
plt.legend(ncol=1,fontsize=9,loc='best')

plt.subplots_adjust( wspace=0.15 , hspace=0.3, top=1.0, left=0, right=0.97, bottom=0.15 )
#plt.tight_layout()



plt.savefig(path0+'batches_ht'+'.eps')





