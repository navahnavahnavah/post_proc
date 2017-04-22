# revived_JDF.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10

plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

col = ['maroon', 'r', 'darkorange', 'gold', 'lawngreen', 'g', 'darkcyan', 'c', 'b', 'navy','purple', 'm', 'hotpink', 'gray', 'k', 'sienna', 'saddlebrown']

secondary = np.array(['', 'stilbite', 'aragonite', 'kaolinite', 'albite', 'saponite_mg', 'celadonite',
'clinoptilolite', 'pyrite', 'mont_na', 'goethite', 'dolomite', 'smectite', 'saponite_k',
'anhydrite', 'siderite', 'calcite', 'quartz', 'kspar', 'saponite_na', 'nont_na', 'nont_mg',
'nont_k', 'fe_celadonite', 'nont_ca', 'muscovite', 'mesolite', 'hematite', 'mont_ca', 'verm_ca',
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

tp = 100
n_box = 3

outpath = "basalt_box_output/mins/batch1/ol1_pyr2_plag1/"

primary_mat5 = np.transpose(np.loadtxt(outpath + 'primary_mat5.txt'))*110.0/2.9
primary_mat4 = np.transpose(np.loadtxt(outpath + 'primary_mat4.txt'))*110.0/2.9

pri_glass = np.transpose(np.loadtxt(outpath + 'primary_mat5.txt'))*110.0/2.9
pri_ol = np.transpose(np.loadtxt(outpath + 'primary_mat4.txt'))*140.0/2.9
pri_pyr = np.transpose(np.loadtxt(outpath + 'primary_mat3.txt'))*140.0/2.9
pri_plag = np.transpose(np.loadtxt(outpath + 'primary_mat2.txt'))*140.0/2.9
primary_mat_total = pri_glass + pri_ol + pri_pyr + pri_plag
pri_glass_dual = pri_glass[0,:] + pri_glass[1,:]
pri_ol_dual = pri_ol[0,:] + pri_ol[1,:]
pri_pyr_dual = pri_pyr[0,:] + pri_pyr[1,:]
pri_plag_dual = pri_plag[0,:] + pri_plag[1,:]

solute_ph = np.transpose(np.loadtxt(outpath + 'solute_ph.txt'))
solute_alk = np.transpose(np.loadtxt(outpath + 'solute_alk.txt'))
solute_w = np.transpose(np.loadtxt(outpath + 'solute_w.txt'))
solute_c = np.transpose(np.loadtxt(outpath + 'solute_c.txt'))
solute_ca = np.transpose(np.loadtxt(outpath + 'solute_ca.txt'))
solute_mg = np.transpose(np.loadtxt(outpath + 'solute_mg.txt'))
solute_na = np.transpose(np.loadtxt(outpath + 'solute_na.txt'))
solute_k = np.transpose(np.loadtxt(outpath + 'solute_k.txt'))
solute_fe = np.transpose(np.loadtxt(outpath + 'solute_fe.txt'))
solute_s = np.transpose(np.loadtxt(outpath + 'solute_s.txt'))
solute_si = np.transpose(np.loadtxt(outpath + 'solute_si.txt'))
solute_cl = np.transpose(np.loadtxt(outpath + 'solute_cl.txt'))
solute_al = np.transpose(np.loadtxt(outpath + 'solute_al.txt'))


solute_ph_mean = np.sum(solute_ph[:-1,:],axis=0)/float(n_box-1)
solute_alk_mean = np.sum(solute_alk[:-1,:],axis=0)/float(n_box-1)
solute_w_mean = np.sum(solute_w[:-1,:],axis=0)/float(n_box-1)
solute_c_mean = np.sum(solute_c[:-1,:],axis=0)/float(n_box-1)
solute_ca_mean = np.sum(solute_ca[:-1,:],axis=0)/float(n_box-1)
solute_mg_mean = np.sum(solute_mg[:-1,:],axis=0)/float(n_box-1)
solute_na_mean = np.sum(solute_na[:-1,:],axis=0)/float(n_box-1)
solute_k_mean = np.sum(solute_k[:-1,:],axis=0)/float(n_box-1)
solute_fe_mean = np.sum(solute_fe[:-1,:],axis=0)/float(n_box-1)
solute_s_mean = np.sum(solute_s[:-1,:],axis=0)/float(n_box-1)
solute_si_mean = np.sum(solute_si[:-1,:],axis=0)/float(n_box-1)
solute_cl_mean = np.sum(solute_cl[:-1,:],axis=0)/float(n_box-1)
solute_al_mean = np.sum(solute_al[:-1,:],axis=0)/float(n_box-1)

secondary_mat = np.zeros([58,n_box,tp])
secondary_mat_vol = np.zeros([58,n_box,tp])

for i in range(1,58):
    if os.path.isfile(outpath + 'secondary_mat' + str(i) + '.txt'):
        secondary_mat[i,:,:] = np.transpose(np.loadtxt(outpath + 'secondary_mat' + str(i) + '.txt'))#*molar[i]/density[i]
secondary_mat_sum = np.sum(secondary_mat[:,:-1,:],axis=1)
print secondary_mat_sum.shape

#times = np.zeros(primary_mat5[0,:])
times = np.linspace(0.0, 1.0, tp)

for i in range(1,58):
    secondary_mat_vol[i,:,:] = secondary_mat[i,:,:]*molar[i]/density[i]
    
alt_vol = np.zeros(secondary_mat_vol[0,:,:].shape)
alt_vol[0,:] = np.sum(secondary_mat_vol[:,0,:], axis=0)
alt_vol[1,:] = np.sum(secondary_mat_vol[:,1,:], axis=0)
alt_vol[2,:] = np.sum(secondary_mat_vol[:,2,:], axis=0)



fig=plt.figure()

ax1=fig.add_subplot(2,1,1)
plt.plot(np.transpose(primary_mat5[0,:]),linestyle='--')
plt.plot(np.transpose(primary_mat5[1,:]),linestyle='-.')
#plt.plot(np.transpose(primary_mat5[2,:]))

ax1=fig.add_subplot(2,1,2)
for i in range(1,58):
    plt.plot(times,np.transpose(secondary_mat[i,:,:]))


plt.savefig(outpath+'mat_primary5'+'.png')


print "all secondary minerals:"
for i in range(1,58):
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        print i , secondary[i]


li = 20



###########################
# CHAMBER BY CHAMBER PLOT #
###########################

mini = 1.0
if np.min(pri_glass[0,:])/np.max(pri_glass[0,:]) < mini:
    mini = np.min(pri_glass[0,:])/np.max(pri_glass[0,:])
if np.min(pri_glass[1,:])/np.max(pri_glass[1,:]) < mini:
    mini = np.min(pri_glass[1,:])/np.max(pri_glass[1,:])
if np.min(pri_glass[2,:])/np.max(pri_glass[2,:]) < mini:
    mini = np.min(pri_glass[2,:])/np.max(pri_glass[2,:])
    
if np.min(pri_ol[0,:])/np.max(pri_ol[0,:]) < mini:
    mini = np.min(pri_ol[0,:])/np.max(pri_ol[0,:])
if np.min(pri_ol[1,:])/np.max(pri_ol[1,:]) < mini:
    mini = np.min(pri_ol[1,:])/np.max(pri_ol[1,:])
if np.min(pri_ol[2,:])/np.max(pri_ol[2,:]) < mini:
    mini = np.min(pri_ol[2,:])/np.max(pri_ol[2,:])
    
if np.min(pri_pyr[0,:])/np.max(pri_pyr[0,:]) < mini:
    mini = np.min(pri_pyr[0,:])/np.max(pri_pyr[0,:])
if np.min(pri_pyr[1,:])/np.max(pri_pyr[1,:]) < mini:
    mini = np.min(pri_pyr[1,:])/np.max(pri_pyr[1,:])
if np.min(pri_pyr[2,:])/np.max(pri_pyr[2,:]) < mini:
    mini = np.min(pri_pyr[2,:])/np.max(pri_pyr[2,:])

if np.min(pri_plag[0,:])/np.max(pri_plag[0,:]) < mini:
    mini = np.min(pri_plag[0,:])/np.max(pri_plag[0,:])
if np.min(pri_plag[1,:])/np.max(pri_plag[1,:]) < mini:
    mini = np.min(pri_plag[1,:])/np.max(pri_plag[1,:])
if np.min(pri_plag[2,:])/np.max(pri_plag[2,:]) < mini:
    mini = np.min(pri_plag[2,:])/np.max(pri_plag[2,:])

fig=plt.figure()

ax1=fig.add_subplot(2,3,1)

plt.plot(times,np.transpose(pri_glass[0,:]/np.max(pri_glass[0,:])),linestyle='-',c='k',lw=1,label='glass')
plt.plot(times,np.transpose(pri_ol[0,:]/np.max(pri_ol[0,:])),linestyle='-',c='limegreen',lw=1,label='olivine')
plt.plot(times,np.transpose(pri_pyr[0,:]/np.max(pri_pyr[0,:])),linestyle='-',c='steelblue',lw=1,label='pyroxene')
plt.plot(times,np.transpose(pri_plag[0,:]/np.max(pri_plag[0,:])),linestyle='-',c='m',lw=1,label='plagioclase')
plt.ylim([mini,1.0])
plt.title('A', x=0.08, y=0.85,horizontalalignment='left')


ax1=fig.add_subplot(2,3,4)

plt.plot(times,np.transpose(pri_ol[1,:]/np.max(pri_ol[1,:])),linestyle='-',c='limegreen',lw=1,label='olivine')
plt.plot(times,np.transpose(pri_pyr[1,:]/np.max(pri_pyr[1,:])),linestyle='-',c='steelblue',lw=1,label='pyroxene')
plt.plot(times,np.transpose(pri_plag[1,:]/np.max(pri_plag[1,:])),linestyle='-',c='m',lw=1,label='plagioclase')
#plt.ylim([mini,1.0])
plt.title('B', x=0.08, y=0.85,horizontalalignment='left')


ax1=fig.add_subplot(1,3,2)

plt.plot(times,np.transpose(pri_glass_dual[:]/np.max(pri_glass_dual[:])),linestyle='-',c='k',lw=1,label='glass')
plt.plot(times,np.transpose(pri_ol_dual[:]/np.max(pri_ol_dual[:])),linestyle='-',c='limegreen',lw=1,label='olivine')
plt.plot(times,np.transpose(pri_pyr_dual[:]/np.max(pri_pyr_dual[:])),linestyle='-',c='steelblue',lw=1,label='pyroxene')
plt.plot(times,np.transpose(pri_plag_dual[:]/np.max(pri_plag_dual[:])),linestyle='-',c='m',lw=1,label='plagioclase')
plt.legend(bbox_to_anchor=(2.2, 1.18), ncol=4, labelspacing=0.0, fontsize=12)
plt.ylim([mini,1.0])
plt.title('A + B', x=0.08, y=0.92,horizontalalignment='left')




ax1=fig.add_subplot(1,3,3)

plt.plot(times,np.transpose(pri_glass[2,:]/np.max(pri_glass[2,:])),linestyle='-',c='k',lw=1,label='glass')
plt.plot(times,np.transpose(pri_ol[2,:]/np.max(pri_ol[2,:])),linestyle='-',c='limegreen',lw=1,label='olivine')
plt.plot(times,np.transpose(pri_pyr[2,:]/np.max(pri_pyr[2,:])),linestyle='-',c='steelblue',lw=1,label='pyroxene')
plt.plot(times,np.transpose(pri_plag[2,:]/np.max(pri_plag[2,:])),linestyle='-',c='m',lw=1,label='plagioclase')
plt.ylim([mini,1.0])
plt.xlabel('time [Myr]')
plt.title('Solo', x=0.08, y=0.92, horizontalalignment='left')


plt.subplots_adjust( wspace=0.4 , hspace=0.2, top=0.85 )

# ax1=fig.add_subplot(2,2,1)
# plt.plot(times,np.transpose(pri_glass[0,:]/np.max(pri_glass[0,:])),linestyle='-',c='k',lw=1,label='glass')
# plt.plot(times,np.transpose(pri_ol[0,:]/np.max(pri_ol[0,:])),linestyle='-',c='lawngreen',lw=1,label='olivine')
# plt.plot(times,np.transpose(pri_pyr[0,:]/np.max(pri_pyr[0,:])),linestyle='-',c='blue',lw=1,label='pyroxene')
# plt.plot(times,np.transpose(pri_plag[0,:]/np.max(pri_plag[0,:])),linestyle='-',c='purple',lw=1,label='plagioclase')
# plt.legend(fontsize=8,loc='best',ncol=1,labelspacing=0.0)
# plt.title('minerals in chamber A')



plt.savefig(outpath+'m_chamberplot'+'.png')

###############
# MASTER PLOT #
###############

fig=plt.figure()

ax1=fig.add_subplot(2,2,1)
plt.plot(times,np.transpose(primary_mat5[2,:]/np.max(primary_mat5[2,:])),linestyle='-',c='c',lw=1,label='Solo chamber, glass')
plt.plot(times,np.transpose((primary_mat5[1,:]+primary_mat5[0,:])/np.max(primary_mat5[1,:]+primary_mat5[0,:])),linestyle='-',c='hotpink',lw=1,label='Dual chamber, glass')
plt.plot(times,np.transpose((primary_mat5[0,:])/np.max(primary_mat5[0,:])),linestyle='--',c='hotpink',lw=1,label='Chamber A')
plt.plot(times,np.transpose((primary_mat5[1,:])/np.max(primary_mat5[1,:])),linestyle=':',c='hotpink',lw=1,label='Chamber B')
plt.xlabel('time [Myr]')
plt.title('Remaining glass fraction')
plt.legend(fontsize=8,loc='best',ncol=1,labelspacing=0.0)

ax1=fig.add_subplot(2,2,2)
plt.plot(times,np.transpose(primary_mat4[2,:]/np.max(primary_mat4[2,:])),linestyle='-',c='c',lw=1,label='Solo chamber, glass')
plt.plot(times,np.transpose((primary_mat4[1,:]+primary_mat4[0,:])/np.max(primary_mat4[1,:]+primary_mat4[0,:])),linestyle='-',c='hotpink',lw=1,label='Dual chamber, glass')
plt.plot(times,np.transpose((primary_mat4[0,:])/np.max(primary_mat4[0,:])),linestyle='--',c='hotpink',lw=1,label='Chamber A')
plt.plot(times,np.transpose((primary_mat4[1,:])/np.max(primary_mat4[1,:])),linestyle=':',c='hotpink',lw=1,label='Chamber B')
plt.xlabel('time [Myr]')
plt.title('Remaining mineral fraction')
plt.legend(fontsize=8,loc='best',ncol=1,labelspacing=0.0)



# plt.plot(times,np.transpose(primary_mat5[0,:]/np.max(primary_mat5[0,:])),linestyle='--',c='c',lw=2,label='Chamber A')
# plt.plot(times,np.transpose(primary_mat5[1,:]/np.max(primary_mat5[1,:])),linestyle=':',c='c',lw=2,label='Chamber B')

ax1=fig.add_subplot(2,2,3)
plt.plot(times,np.transpose(alt_vol[2,:]/(alt_vol[2,:]+primary_mat_total[2,:])),linestyle='-',c='hotpink',lw=3,label='Solo chamber')
plt.plot(times,np.transpose((alt_vol[1,:]+alt_vol[0,:])/(alt_vol[1,:]+alt_vol[0,:]+primary_mat_total[1,:]+primary_mat_total[0,:])),linestyle='-',c='c',lw=3,label='Dual chamber')
plt.plot(times,np.transpose(alt_vol[0,:]/(alt_vol[0,:]+primary_mat_total[0,:])),linestyle='--',c='c',lw=2,label='Chamber A')
plt.plot(times,np.transpose(alt_vol[1,:]/(alt_vol[1,:]+primary_mat_total[0,:])),linestyle=':',c='c',lw=2,label='Chamber B')
plt.xlabel('time [Myr]')
plt.title('Fractional alteration volume')
plt.legend(fontsize=8,loc='best',ncol=1,labelspacing=0.0)


plt.subplots_adjust( wspace=0.2 , hspace=0.3)
plt.savefig(outpath+'masterplot'+'.png')





####################
# SECONDARY TRIPLE #
####################

fig=plt.figure()

ax1=fig.add_subplot(2,3,1)

j = 0
for i in range(1,58):
    if np.max(secondary_mat[i,:,:]) > 0.0:
        print j
        if np.max(secondary_mat[i,0,:]) > 0.0:
            plt.plot(times,np.transpose(secondary_mat[i,0,:]), c=col[j], label=secondary[i], lw=2)
        j = j+1
#plt.legend(fontsize=8,loc='best',ncol=2)
#plt.xlabel('time [Myr]')
plt.title('A', x=0.08, y=0.85,horizontalalignment='left')


ax1=fig.add_subplot(2,3,4)

j = 0
for i in range(1,58):
    if np.max(secondary_mat[i,:,:]) > 0.0:
        if np.max(secondary_mat[i,1,:]) > 0.0:
            plt.plot(times,np.transpose(secondary_mat[i,1,:]), c=col[j], label=secondary[i], lw=2)
        j = j+1
#plt.legend(fontsize=8,loc='best',ncol=2)
plt.xlabel('time [Myr]')
plt.title('B', x=0.08, y=0.85,horizontalalignment='left')


ax1=fig.add_subplot(1,3,2)

j = 0
for i in range(1,58):
    if np.max(secondary_mat[i,:,:]) > 0.0:
        if np.max(secondary_mat_sum[i,:]) > 0.0:
            plt.plot(times,np.transpose(secondary_mat_sum[i,:]), c=col[j], label=secondary[i], lw=2)
        j = j+1
plt.legend(bbox_to_anchor=(0.4, 1.18), fontsize=8,ncol=3, labelspacing=0.0)
plt.xlabel('time [Myr]')
plt.title('A + B', x=0.08, y=0.92,horizontalalignment='left')




ax1=fig.add_subplot(1,3,3)

j = 0
for i in range(1,58):
    if np.max(secondary_mat[i,:,:]) > 0.0:
        if np.max(secondary_mat[i,2,:]) > 0.0:
            plt.plot(times,np.transpose(secondary_mat[i,2,:]), c=col[j], label=secondary[i], lw=2)
        j = j+1
#plt.legend(fontsize=8,loc='best',ncol=2)
plt.legend(bbox_to_anchor=(1.3, 1.18), fontsize=8,ncol=3, labelspacing=0.0)
plt.xlabel('time [Myr]')
plt.title('Solo', x=0.08, y=0.92, horizontalalignment='left')


plt.subplots_adjust( wspace=0.2 , hspace=0.2, top=0.85 )
plt.savefig(outpath+'mat_secondary_triple'+'.png')






##################
# SECONDARY FULL #
##################

fig=plt.figure()

ax1=fig.add_subplot(3,2,1)
j = 0
for i in [7, 26, 30, 34, 38, 44, 39, 31]:
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        plt.plot(np.transpose(secondary_mat_sum[i,:]), c=col[j], linestyle="-", lw=2, label=secondary[i])
        plt.plot(np.transpose(secondary_mat[i,0,:]), c=col[j], linestyle="--")
        plt.plot(np.transpose(secondary_mat[i,1,:]), c=col[j], linestyle=":")
        plt.plot(np.transpose(secondary_mat[i,2,:]), c=col[j], linestyle="-")
        j = j + 1
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('zeolites')

ax1=fig.add_subplot(3,2,2)
j = 0
for i in [9, 28, 56, 57, 5, 13, 19, 20, 21, 22, 23, 24, 48, 12, 41]:
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        plt.plot(np.transpose(secondary_mat_sum[i,:]), c=col[j], linestyle="-", lw=2, label=secondary[i])
        plt.plot(np.transpose(secondary_mat[i,0,:]), c=col[j], linestyle="--")
        plt.plot(np.transpose(secondary_mat[i,1,:]), c=col[j], linestyle=":")
        plt.plot(np.transpose(secondary_mat[i,2,:]), c=col[j], linestyle="-")
        j = j + 1
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('smectites')

ax1=fig.add_subplot(3,2,3)
j = 0
for i in [43, 45, 46, 47, 53, 54]:
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        plt.plot(np.transpose(secondary_mat_sum[i,:]), c=col[j], linestyle="-", lw=2, label=secondary[i])
        plt.plot(np.transpose(secondary_mat[i,0,:]), c=col[j], linestyle="--")
        plt.plot(np.transpose(secondary_mat[i,1,:]), c=col[j], linestyle=":")
        plt.plot(np.transpose(secondary_mat[i,2,:]), c=col[j], linestyle="-")
        j = j + 1
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('chlorites')

ax1=fig.add_subplot(3,2,4)
j = 0
for i in [1, 2, 3, 4, 6, 8, 10, 11, 14, 15, 17, 18, 25, 27, 32, 33, 36, 40, 42, 50, 51, 52]:
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        plt.plot(np.transpose(secondary_mat_sum[i,:]), c=col[j], linestyle="-", lw=2, label=secondary[i])
        plt.plot(np.transpose(secondary_mat[i,0,:]), c=col[j], linestyle="--")
        plt.plot(np.transpose(secondary_mat[i,1,:]), c=col[j], linestyle=":")
        plt.plot(np.transpose(secondary_mat[i,2,:]), c=col[j], linestyle="-")
        j = j + 1
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('others')

ax1=fig.add_subplot(3,2,5)
j = 0
for i in [16]:
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        plt.plot(np.transpose(secondary_mat_sum[i,:]), c=col[j], linestyle="-", lw=2, label=secondary[i])
        plt.plot(np.transpose(secondary_mat[i,0,:]), c=col[j], linestyle="--")
        plt.plot(np.transpose(secondary_mat[i,1,:]), c=col[j], linestyle=":")
        plt.plot(np.transpose(secondary_mat[i,2,:]), c=col[j], linestyle="-")
        j = j + 1
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('carbonates')

plt.subplots_adjust( wspace=0.2 , hspace=0.3 )
plt.savefig(outpath+'mat_secondary'+'.png')


##################
# SECONDARY INIT #
##################



fig=plt.figure()

ax1=fig.add_subplot(3,2,1)
j = 0
for i in [7, 26, 30, 34, 38, 44, 39, 31]:
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        plt.plot(np.transpose(secondary_mat_sum[i,:li]), c=col[j], linestyle="-", lw=2, label=secondary[i])
        plt.plot(np.transpose(secondary_mat[i,0,:li]), c=col[j], linestyle="--")
        plt.plot(np.transpose(secondary_mat[i,1,:li]), c=col[j], linestyle=":")
        plt.plot(np.transpose(secondary_mat[i,2,:li]), c=col[j], linestyle="-")
        j = j + 1
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('zeolites')

ax1=fig.add_subplot(3,2,2)
j = 0
for i in [9, 28, 56, 57, 5, 13, 19, 20, 21, 22, 23, 24, 48, 12, 41]:
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        plt.plot(np.transpose(secondary_mat_sum[i,:li]), c=col[j], linestyle="-", lw=2, label=secondary[i])
        plt.plot(np.transpose(secondary_mat[i,0,:li]), c=col[j], linestyle="--")
        plt.plot(np.transpose(secondary_mat[i,1,:li]), c=col[j], linestyle=":")
        plt.plot(np.transpose(secondary_mat[i,2,:li]), c=col[j], linestyle="-")
        j = j + 1
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('smectites')

ax1=fig.add_subplot(3,2,3)
j = 0
for i in [43, 45, 46, 47, 53, 54]:
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        plt.plot(np.transpose(secondary_mat_sum[i,:li]), c=col[j], linestyle="-", lw=2, label=secondary[i])
        plt.plot(np.transpose(secondary_mat[i,0,:li]), c=col[j], linestyle="--")
        plt.plot(np.transpose(secondary_mat[i,1,:li]), c=col[j], linestyle=":")
        plt.plot(np.transpose(secondary_mat[i,2,:li]), c=col[j], linestyle="-")
        j = j + 1
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('chlorites')

ax1=fig.add_subplot(3,2,4)
j = 0
for i in [1, 2, 3, 4, 6, 8, 10, 11, 14, 15, 17, 18, 25, 27, 32, 33, 36, 40, 42, 50, 51, 52]:
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        plt.plot(np.transpose(secondary_mat_sum[i,:li]), c=col[j], linestyle="-", lw=2, label=secondary[i])
        plt.plot(np.transpose(secondary_mat[i,0,:li]), c=col[j], linestyle="--")
        plt.plot(np.transpose(secondary_mat[i,1,:li]), c=col[j], linestyle=":")
        plt.plot(np.transpose(secondary_mat[i,2,:li]), c=col[j], linestyle="-")
        j = j + 1
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('others')

ax1=fig.add_subplot(3,2,5)
j = 0
for i in [16]:
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        plt.plot(np.transpose(secondary_mat_sum[i,:li]), c=col[j], linestyle="-", lw=2, label=secondary[i])
        plt.plot(np.transpose(secondary_mat[i,0,:li]), c=col[j], linestyle="--")
        plt.plot(np.transpose(secondary_mat[i,1,:li]), c=col[j], linestyle=":")
        plt.plot(np.transpose(secondary_mat[i,2,:li]), c=col[j], linestyle="-")
        j = j + 1
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('carbonates')

plt.subplots_adjust( wspace=0.2 , hspace=0.3 )
plt.savefig(outpath+'mat_secondary_init'+'.png')





###############
# SOLUTE FULL #
###############

fig=plt.figure()

ax1=fig.add_subplot(3,2,1)
plt.plot(times[li:],np.transpose(solute_ph_mean[li:]), c=col[0], linestyle="-", lw=2, label='pH')
plt.plot(times[li:],np.transpose(solute_ph[0,li:]), c=col[0], linestyle="--")
plt.plot(times[li:],np.transpose(solute_ph[1,li:]), c=col[0], linestyle="-.")
plt.plot(times[li:],np.transpose(solute_ph[2,li:]), c=col[0], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


ax1=fig.add_subplot(3,2,2)
plt.plot(times[li:],np.transpose(solute_alk_mean[li:]), c=col[0], linestyle="-", lw=2, label='alk')
plt.plot(times[li:],np.transpose(solute_alk[0,li:]), c=col[0], linestyle="--")
plt.plot(times[li:],np.transpose(solute_alk[1,li:]), c=col[0], linestyle="-.")
plt.plot(times[li:],np.transpose(solute_alk[2,li:]), c=col[0], linestyle="-")

plt.plot(times[li:],np.transpose(solute_c_mean[li:]), c=col[1], linestyle="-", lw=2, label='dic')
plt.plot(times[li:],np.transpose(solute_c[0,li:]), c=col[1], linestyle="--")
plt.plot(times[li:],np.transpose(solute_c[1,li:]), c=col[1], linestyle="-.")
plt.plot(times[li:],np.transpose(solute_c[2,li:]), c=col[1], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


ax1=fig.add_subplot(3,2,3)
plt.plot(times[li:],np.transpose(solute_ca_mean[li:]), c=col[0], linestyle="-", lw=2, label='ca')
plt.plot(times[li:],np.transpose(solute_ca[0,li:]), c=col[0], linestyle="--")
plt.plot(times[li:],np.transpose(solute_ca[1,li:]), c=col[0], linestyle="-.")
plt.plot(times[li:],np.transpose(solute_ca[2,li:]), c=col[0], linestyle="-")

plt.plot(times[li:],np.transpose(solute_mg_mean[li:]), c=col[1], linestyle="-", lw=2, label='mg')
plt.plot(times[li:],np.transpose(solute_mg[0,li:]), c=col[1], linestyle="--")
plt.plot(times[li:],np.transpose(solute_mg[1,li:]), c=col[1], linestyle="-.")
plt.plot(times[li:],np.transpose(solute_mg[2,li:]), c=col[1], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


ax1=fig.add_subplot(3,2,4)
plt.plot(times[li:],np.transpose(solute_k_mean[li:]), c=col[0], linestyle="-", lw=2, label='k')
plt.plot(times[li:],np.transpose(solute_k[0,li:]), c=col[0], linestyle="--")
plt.plot(times[li:],np.transpose(solute_k[1,li:]), c=col[0], linestyle="-.")
plt.plot(times[li:],np.transpose(solute_k[2,li:]), c=col[0], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


ax1=fig.add_subplot(3,2,5)
plt.plot(times[li:],np.transpose(solute_si_mean[li:]), c=col[0], linestyle="-", lw=2, label='si')
plt.plot(times[li:],np.transpose(solute_si[0,li:]), c=col[0], linestyle="--")
plt.plot(times[li:],np.transpose(solute_si[1,li:]), c=col[0], linestyle="-.")
plt.plot(times[li:],np.transpose(solute_si[2,li:]), c=col[0], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


ax1=fig.add_subplot(3,2,6)
plt.plot(times[li:],np.transpose(solute_al_mean[li:]), c=col[0], linestyle="-", lw=2, label='al')
plt.plot(times[li:],np.transpose(solute_al[0,li:]), c=col[0], linestyle="--")
plt.plot(times[li:],np.transpose(solute_al[1,li:]), c=col[0], linestyle="-.")
plt.plot(times[li:],np.transpose(solute_al[2,li:]), c=col[0], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


plt.subplots_adjust( wspace=0.3 , hspace=0.2 )
plt.savefig(outpath+'mat_solute'+'.png')





###############
# SOLUTE INIT #
###############

fig=plt.figure()

ax1=fig.add_subplot(3,2,1)
plt.plot(np.transpose(solute_ph_mean[:li]), c=col[0], linestyle="-", lw=2, label='pH')
plt.plot(np.transpose(solute_ph[0,:li]), c=col[0], linestyle="--")
plt.plot(np.transpose(solute_ph[1,:li]), c=col[0], linestyle="-.")
plt.plot(np.transpose(solute_ph[2,:li]), c=col[0], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


ax1=fig.add_subplot(3,2,2)
plt.plot(np.transpose(solute_alk_mean[:li]), c=col[0], linestyle="-", lw=2, label='alk')
plt.plot(np.transpose(solute_alk[0,:li]), c=col[0], linestyle="--")
plt.plot(np.transpose(solute_alk[1,:li]), c=col[0], linestyle="-.")
plt.plot(np.transpose(solute_alk[2,:li]), c=col[0], linestyle="-")

plt.plot(np.transpose(solute_c_mean[:li]), c=col[1], linestyle="-", lw=2, label='dic')
plt.plot(np.transpose(solute_c[0,:li]), c=col[1], linestyle="--")
plt.plot(np.transpose(solute_c[1,:li]), c=col[1], linestyle="-.")
plt.plot(np.transpose(solute_c[2,:li]), c=col[1], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


ax1=fig.add_subplot(3,2,3)
plt.plot(np.transpose(solute_ca_mean[:li]), c=col[0], linestyle="-", lw=2, label='ca')
plt.plot(np.transpose(solute_ca[0,:li]), c=col[0], linestyle="--")
plt.plot(np.transpose(solute_ca[1,:li]), c=col[0], linestyle="-.")
plt.plot(np.transpose(solute_ca[2,:li]), c=col[0], linestyle="-")

plt.plot(np.transpose(solute_mg_mean[:li]), c=col[1], linestyle="-", lw=2, label='mg')
plt.plot(np.transpose(solute_mg[0,:li]), c=col[1], linestyle="--")
plt.plot(np.transpose(solute_mg[1,:li]), c=col[1], linestyle="-.")
plt.plot(np.transpose(solute_mg[2,:li]), c=col[1], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


ax1=fig.add_subplot(3,2,4)
plt.plot(np.transpose(solute_k_mean[:li]), c=col[0], linestyle="-", lw=2, label='k')
plt.plot(np.transpose(solute_k[0,:li]), c=col[0], linestyle="--")
plt.plot(np.transpose(solute_k[1,:li]), c=col[0], linestyle="-.")
plt.plot(np.transpose(solute_k[2,:li]), c=col[0], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


ax1=fig.add_subplot(3,2,5)
plt.plot(np.transpose(solute_si_mean[:li]), c=col[0], linestyle="-", lw=2, label='si')
plt.plot(np.transpose(solute_si[0,:li]), c=col[0], linestyle="--")
plt.plot(np.transpose(solute_si[1,:li]), c=col[0], linestyle="-.")
plt.plot(np.transpose(solute_si[2,:li]), c=col[0], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


ax1=fig.add_subplot(3,2,6)
plt.plot(np.transpose(solute_al_mean[:li]), c=col[0], linestyle="-", lw=2, label='al')
plt.plot(np.transpose(solute_al[0,:li]), c=col[0], linestyle="--")
plt.plot(np.transpose(solute_al[1,:li]), c=col[0], linestyle="-.")
plt.plot(np.transpose(solute_al[2,:li]), c=col[0], linestyle="-")
plt.legend(fontsize=8,loc='best',ncol=2)


plt.subplots_adjust( wspace=0.3 , hspace=0.2 )
plt.savefig(outpath+'mat_solute_init'+'.png')


