# plot_basalt_box.py

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

# col = ['maroon', 'r', 'darkorange', 'gold', 'lawngreen', 'g', 'darkcyan', 'c', 'b', 'navy','purple', 'm', 'hotpink', 'gray', 'k', 'sienna', 'saddlebrown']

col = ['#6e0202', '#fc385b', '#ff7411', '#19a702', '#00520d', '#00ffc2', '#609ff2', '#20267c','#8f00ff', '#ec52ff', '#6e6e6e', '#000000', '#c6813a', '#7d4e22', '#ffff00', '#df9a00', '#812700', '#6b3f67', '#0f9995', '#4d4d4d', '#d9d9d9', '#e9acff']

secondary = np.array(['', 'kaolinite', 'saponite_mg', 'celadonite', 'clinoptilolite', 'pyrite', 'mont_na', 'goethite',
'smectite', 'calcite', 'kspar', 'saponite_na', 'nont_na', 'nont_mg', 'fe_celad', 'nont_ca',
'mesolite', 'hematite', 'mont_ca', 'verm_ca', 'analcime', 'philipsite', 'mont_mg', 'gismondine',
'verm_mg', 'natrolite', 'talc', 'smectite_low', 'prehnite', 'chlorite', 'scolecite', 'clinochlorte14a',
'clinochlore7a', 'saponite_ca', 'verm_na', 'pyrrhotite', 'fe_saponite_ca', 'fe_saponite_mg', 'daphnite7a', 'daphnite14a', 'epidote'])

density = np.array([0.0, 2.65, 2.3, 3.05, 2.17, 5.01, 2.5, 3.8,
2.7, 2.71, 2.56, 2.3, 2.28, 2.28, 3.05, 2.28,
2.25, 5.3, 2.5, 2.55, 2.27, 2.2, 2.5, 2.26,
2.55, 2.25, 2.75, 2.7, 2.87, 2.9, 2.275, 2.8,
2.8, 2.3, 2.55, 4.61, 2.3, 2.3, 3.2, 3.2, 3.45])

molar = np.array([0.0, 258.156, 480.19, 429.02, 2742.13, 119.98, 549.07, 88.851,
549.07, 100.0869, 287.327, 480.19, 495.90, 495.90, 429.02, 495.90,
380.22, 159.6882, 549.07, 504.19, 220.15, 649.86, 549.07, 649.86,
504.19, 380.22, 379.259, 549.07, 395.38, 64.448, 392.34, 64.448,
64.448, 480.19, 504.19, 85.12, 480.19, 480.19, 664.0, 664.0, 519.0])

tp = 100
n_box = 3
minNum = 41

#todo: path here
outpath = "../output/revival/winter_basalt_box/jj_mix_1e11/"

primary_mat5 = np.transpose(np.loadtxt(outpath + 'z_primary_mat5.txt'))*110.0/2.7
primary_mat4 = np.transpose(np.loadtxt(outpath + 'z_primary_mat4.txt'))*110.0/2.7

pri_glass = np.transpose(np.loadtxt(outpath + 'z_primary_mat5.txt'))*110.0/2.7
pri_ol = np.transpose(np.loadtxt(outpath + 'z_primary_mat4.txt'))*140.0/2.7
pri_pyr = np.transpose(np.loadtxt(outpath + 'z_primary_mat3.txt'))*140.0/2.7
pri_plag = np.transpose(np.loadtxt(outpath + 'z_primary_mat2.txt'))*140.0/2.7
primary_mat_total = pri_glass + pri_ol + pri_pyr + pri_plag
pri_glass_dual = pri_glass[1,:] + pri_glass[2,:]


solute_ph = np.transpose(np.loadtxt(outpath + 'z_solute_ph.txt'))
solute_alk = np.transpose(np.loadtxt(outpath + 'z_solute_alk.txt'))
solute_w = np.transpose(np.loadtxt(outpath + 'z_solute_w.txt'))
solute_c = np.transpose(np.loadtxt(outpath + 'z_solute_c.txt'))
solute_ca = np.transpose(np.loadtxt(outpath + 'z_solute_ca.txt'))
solute_mg = np.transpose(np.loadtxt(outpath + 'z_solute_mg.txt'))
solute_na = np.transpose(np.loadtxt(outpath + 'z_solute_na.txt'))
solute_k = np.transpose(np.loadtxt(outpath + 'z_solute_k.txt'))
solute_fe = np.transpose(np.loadtxt(outpath + 'z_solute_fe.txt'))
solute_s = np.transpose(np.loadtxt(outpath + 'z_solute_s.txt'))
solute_si = np.transpose(np.loadtxt(outpath + 'z_solute_si.txt'))
solute_cl = np.transpose(np.loadtxt(outpath + 'z_solute_cl.txt'))
solute_al = np.transpose(np.loadtxt(outpath + 'z_solute_al.txt'))


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

secondary_mat = np.zeros([minNum+1,n_box,tp])
secondary_mat_vol = np.zeros([minNum+1,n_box,tp])

for i in range(1,minNum):
    if os.path.isfile(outpath + 'z_secondary_mat' + str(i) + '.txt'):
        secondary_mat[i,:,:] = np.transpose(np.loadtxt(outpath + 'z_secondary_mat' + str(i) + '.txt'))#*molar[i]/density[i]
secondary_mat_sum = np.sum(secondary_mat[:,1:,:],axis=1)
print secondary_mat_sum.shape

#times = np.zeros(primary_mat5[0,:])
times = np.linspace(0.0, 100.0, tp)

for i in range(1,minNum):
    secondary_mat_vol[i,:,:] = secondary_mat[i,:,:]*molar[i]/density[i]

alt_vol = np.zeros(secondary_mat_vol[0,:,:].shape)
alt_vol[0,:] = np.sum(secondary_mat_vol[:,0,:], axis=0)
alt_vol[1,:] = np.sum(secondary_mat_vol[:,1,:], axis=0)
alt_vol[2,:] = np.sum(secondary_mat_vol[:,2,:], axis=0)



# fig=plt.figure()
#
# ax1=fig.add_subplot(2,1,1)
# plt.plot(np.transpose(primary_mat5[0,:]),linestyle='--')
# plt.plot(np.transpose(primary_mat5[1,:]),linestyle='-.')
# #plt.plot(np.transpose(primary_mat5[2,:]))
#
# ax1=fig.add_subplot(2,1,2)
# for i in range(1,minNum):
#     plt.plot(times,np.transpose(secondary_mat[i,:,:]))
#
#
# plt.savefig(outpath+'mat_primary5'+'.png',bbox_inches='tight')


print "all secondary minerals:"
for i in range(1,minNum):
    if np.max(secondary_mat_sum[i,:]) > 0.0:
        print i , secondary[i]


li = 20


#hack: sol plot

fig=plt.figure(figsize=(6,6))

plt.plot(times,solute_w[0,:],c='#ab250d',lw=1,label='s')
plt.plot(times,solute_w[1,:],c='#c39e31',lw=1,label='a')
plt.plot(times,solute_w[2,:],c='#138b0d',lw=1,label='b')


plt.savefig(outpath+'m_sol_w'+'.png',bbox_inches='tight')




#hack: m_ chamber plot

mini = 1.0
if np.min(pri_glass[0,:])/np.max(pri_glass[0,:]) < mini:
    mini = np.min(pri_glass[0,:])/np.max(pri_glass[0,:])
if np.min(pri_glass[1,:])/np.max(pri_glass[1,:]) < mini:
    mini = np.min(pri_glass[1,:])/np.max(pri_glass[1,:])
if np.min(pri_glass[2,:])/np.max(pri_glass[2,:]) < mini:
    mini = np.min(pri_glass[2,:])/np.max(pri_glass[2,:])





fig=plt.figure(figsize=(12,7))

ax1=fig.add_subplot(2,3,1)

plt.plot(times,np.transpose(pri_glass[1,:]/np.max(pri_glass[1,:])),linestyle='-',c='k',lw=1,label='glass')
plt.ylim([mini,1.0])
plt.title('A', x=0.08, y=0.85,horizontalalignment='left')


ax1=fig.add_subplot(2,3,4)

plt.plot(times,np.transpose(pri_glass[2,:]/np.max(pri_glass[2,:])),linestyle='-',c='k',lw=1,label='glass')
#plt.ylim([mini,1.0])
plt.title('B', x=0.08, y=0.85,horizontalalignment='left')


ax1=fig.add_subplot(1,3,2)

plt.plot(times,np.transpose(pri_glass_dual[:]/np.max(pri_glass_dual[:])),linestyle='-',c='k',lw=1,label='glass')
#plt.legend(bbox_to_anchor=(2.2, 1.18), ncol=4, labelspacing=0.0, fontsize=12)
plt.ylim([mini,1.0])
plt.title('A + B', x=0.08, y=0.92,horizontalalignment='left')




ax1=fig.add_subplot(1,3,3)

plt.plot(times,np.transpose(pri_glass[0,:]/np.max(pri_glass[0,:])),linestyle='-',c='k',lw=1,label='glass')
plt.ylim([mini,1.0])
plt.xlabel('time [Myr]')
plt.title('Solo', x=0.08, y=0.92, horizontalalignment='left')


plt.subplots_adjust( wspace=0.4 , hspace=0.2, top=0.85 )



plt.savefig(outpath+'m_chamberplot'+'.png',bbox_inches='tight')



#hack: master plot

fig=plt.figure(figsize=(10.0,5.0))

ax1=fig.add_subplot(1,2,1)
plt.plot(times,np.transpose(primary_mat5[0,:]/np.max(primary_mat5[0,:])),linestyle='-',c='c',lw=1,label='Solo chamber, glass')
plt.plot(times,np.transpose((primary_mat5[1,:]+primary_mat5[2,:])/np.max(primary_mat5[1,:]+primary_mat5[2,:])),linestyle='-',c='hotpink',lw=1,label='Dual chamber, glass')
plt.plot(times,np.transpose((primary_mat5[1,:])/np.max(primary_mat5[1,:])),linestyle='--',c='orange',lw=1,label='Chamber A')
plt.plot(times,np.transpose((primary_mat5[2,:])/np.max(primary_mat5[2,:])),linestyle=':',c='green',lw=1,label='Chamber B')
plt.xlabel('time [Myr]')
plt.title('Remaining glass fraction')
plt.legend(fontsize=8,loc='best',ncol=1,labelspacing=0.0)



ax1=fig.add_subplot(1,2,2)
plt.plot(times,np.transpose(alt_vol[0,:]/(alt_vol[0,:]+primary_mat_total[0,:])),linestyle='-',c='hotpink',lw=3,label='Solo chamber')
plt.plot(times,np.transpose((alt_vol[1,:]+alt_vol[2,:])/(alt_vol[1,:]+alt_vol[2,:]+primary_mat_total[1,:]+primary_mat_total[2,:])),linestyle='-',c='c',lw=3,label='Dual chamber')
plt.plot(times,np.transpose(alt_vol[1,:]/(alt_vol[1,:]+primary_mat_total[1,:])),linestyle='--',c='orange',lw=2,label='Chamber A')
plt.plot(times,np.transpose(alt_vol[2,:]/(alt_vol[2,:]+primary_mat_total[2,:])),linestyle=':',c='green',lw=2,label='Chamber B')
plt.xlabel('time [Myr]')
plt.title('Fractional alteration volume')
plt.legend(fontsize=8,loc='best',ncol=1,labelspacing=0.0)


plt.subplots_adjust( wspace=0.2 , hspace=0.3)
plt.savefig(outpath+'masterplot'+'.png',bbox_inches='tight')





#hack: secondary triple

fig=plt.figure()

the_lw = 1.0

ax1=fig.add_subplot(2,3,1)

j = 0
for i in range(1,minNum):
    if np.max(secondary_mat[i,:,:]) > 0.0:
        print j
        if np.max(secondary_mat[i,1,:]) > 0.0:
            plt.plot(times,np.transpose(secondary_mat[i,1,:]), c=col[j], label=secondary[i], lw=the_lw)
        j = j+1
#plt.legend(fontsize=8,loc='best',ncol=2)
#plt.xlabel('time [Myr]')
plt.title('A', x=0.08, y=0.85,horizontalalignment='left')


ax1=fig.add_subplot(2,3,4)

j = 0
for i in range(1,minNum):
    if np.max(secondary_mat[i,:,:]) > 0.0:
        if np.max(secondary_mat[i,2,:]) > 0.0:
            plt.plot(times,np.transpose(secondary_mat[i,2,:]), c=col[j], label=secondary[i], lw=the_lw)
        j = j+1
#plt.legend(fontsize=8,loc='best',ncol=2)
plt.xlabel('time [Myr]')
plt.title('B', x=0.08, y=0.85,horizontalalignment='left')


ax1=fig.add_subplot(1,3,2)

j = 0
for i in range(1,minNum):
    if np.max(secondary_mat[i,:,:]) > 0.0:
        if np.max(secondary_mat_sum[i,:]) > 0.0:
            plt.plot(times,np.transpose(secondary_mat_sum[i,:]), c=col[j], label=secondary[i], lw=the_lw)
        j = j+1
plt.legend(bbox_to_anchor=(0.4, 1.18), fontsize=8,ncol=3, labelspacing=0.0)
plt.xlabel('time [Myr]')
plt.title('A + B', x=0.08, y=0.92,horizontalalignment='left')




ax1=fig.add_subplot(1,3,3)

j = 0
for i in range(1,minNum):
    if np.max(secondary_mat[i,:,:]) > 0.0:
        if np.max(secondary_mat[i,0,:]) > 0.0:
            plt.plot(times,np.transpose(secondary_mat[i,0,:]), c=col[j], label=secondary[i], lw=the_lw)
        j = j+1
#plt.legend(fontsize=8,loc='best',ncol=2)
plt.legend(bbox_to_anchor=(1.3, 1.18), fontsize=8,ncol=3, labelspacing=0.0)
plt.xlabel('time [Myr]')
plt.title('Solo', x=0.08, y=0.92, horizontalalignment='left')


plt.subplots_adjust( wspace=0.2 , hspace=0.2, top=0.85 )
plt.savefig(outpath+'mat_secondary_triple'+'.png',bbox_inches='tight')



#hack SOLUTE STUFF
the_lw = 1.5
the_fs = 8

kwargs_s = dict(color='#7a1807', linewidth=the_lw, label='s')
kwargs_d = dict(color='#2a49b4', linewidth=the_lw, label='d')
kwargs_a = dict(color='#2b90d1', linewidth=the_lw, label='a')
kwargs_b = dict(color='#2ab455', linewidth=the_lw, label='b')



#hack: solute full

fig=plt.figure(figsize=(12.0,4.0))

li = 0

ax1=fig.add_subplot(2,4,1)
# plt.plot(times[li:],np.transpose(solute_ph_mean[li:]), **kwargs_s)
plt.plot(times[li:],np.transpose(solute_ph[0,li:]), **kwargs_s)
plt.plot(times[li:],np.transpose(solute_ph[1,li:]), **kwargs_a)
plt.plot(times[li:],np.transpose(solute_ph[2,li:]), **kwargs_b)
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('pH', fontsize=the_fs)


ax1=fig.add_subplot(2,4,2)
#plt.plot(times[li:],np.transpose(solute_alk_mean[li:]),  **kwargs_s)
plt.plot(times[li:],np.transpose(solute_alk[0,li:]), **kwargs_s)
plt.plot(times[li:],np.transpose(solute_alk[1,li:]), **kwargs_a)
plt.plot(times[li:],np.transpose(solute_alk[2,li:]), **kwargs_b)
plt.title('alk', fontsize=the_fs)


ax1=fig.add_subplot(2,4,3)
#plt.plot(times[li:],np.transpose(solute_c_mean[li:]), c=col[1], linestyle="-", lw=2, label='dic')
plt.plot(times[li:],np.transpose(solute_c[0,li:]), **kwargs_s)
#plt.plot(times[li:],np.transpose(solute_c[1,li:]), **kwargs_a)
#plt.plot(times[li:],np.transpose(solute_c[2,li:]), **kwargs_b)
plt.title('C', fontsize=the_fs)


ax1=fig.add_subplot(2,4,4)
#plt.plot(times[li:],np.transpose(solute_ca_mean[li:]), c=col[0], linestyle="-", lw=2, label='ca')
plt.plot(times[li:],np.transpose(solute_ca[0,li:]), **kwargs_s)
plt.plot(times[li:],np.transpose(solute_ca[1,li:]), **kwargs_a)
plt.plot(times[li:],np.transpose(solute_ca[2,li:]), **kwargs_b)
plt.title('Ca', fontsize=the_fs)


ax1=fig.add_subplot(2,4,5)
#plt.plot(times[li:],np.transpose(solute_mg_mean[li:]), **kwargs_s)
plt.plot(times[li:],np.transpose(solute_mg[0,li:]), **kwargs_s)
plt.plot(times[li:],np.transpose(solute_mg[1,li:]), **kwargs_a)
plt.plot(times[li:],np.transpose(solute_mg[2,li:]), **kwargs_b)
plt.title('Mg', fontsize=the_fs)



ax1=fig.add_subplot(2,4,6)
#plt.plot(times[li:],np.transpose(solute_k_mean[li:]), c=col[0], linestyle="-", lw=2, label='k')
plt.plot(times[li:],np.transpose(solute_k[0,li:]), **kwargs_s)
plt.plot(times[li:],np.transpose(solute_k[1,li:]), **kwargs_a)
plt.plot(times[li:],np.transpose(solute_k[2,li:]), **kwargs_b)
plt.title('K', fontsize=the_fs)



ax1=fig.add_subplot(2,4,7)
#plt.plot(times[li:],np.transpose(solute_si_mean[li:]), c=col[0], linestyle="-", lw=2, label='si')
plt.plot(times[li:],np.transpose(solute_si[0,li:]), **kwargs_s)
plt.plot(times[li:],np.transpose(solute_si[1,li:]), **kwargs_a)
plt.plot(times[li:],np.transpose(solute_si[2,li:]), **kwargs_b)
plt.title('Si', fontsize=the_fs)



ax1=fig.add_subplot(2,4,8)
#plt.plot(times[li:],np.transpose(solute_al_mean[li:]), c=col[0], linestyle="-", lw=2, label='al')
plt.plot(times[li:],np.transpose(solute_al[0,li:]), **kwargs_s)
plt.plot(times[li:],np.transpose(solute_al[1,li:]), **kwargs_a)
plt.plot(times[li:],np.transpose(solute_al[2,li:]), **kwargs_b)
plt.title('Al', fontsize=the_fs)



plt.subplots_adjust( wspace=0.5 , hspace=0.2 )
plt.savefig(outpath+'mat_solute_full'+'.png',bbox_inches='tight')





#hack: solute init


li = 20

fig=plt.figure(figsize=(12.0,4.0))



ax1=fig.add_subplot(2,4,1)
# plt.plot(times[:li],np.transpose(solute_ph_mean[:li]), **kwargs_s)
plt.plot(times[:li],np.transpose(solute_ph[0,:li]), **kwargs_s)
plt.plot(times[:li],np.transpose(solute_ph[1,:li]), **kwargs_a)
plt.plot(times[:li],np.transpose(solute_ph[2,:li]), **kwargs_b)
plt.legend(fontsize=8,loc='best',ncol=2)
plt.title('pH', fontsize=the_fs)


ax1=fig.add_subplot(2,4,2)
#plt.plot(times[:li],np.transpose(solute_alk_mean[:li]),  **kwargs_s)
plt.plot(times[:li],np.transpose(solute_alk[0,:li]), **kwargs_s)
plt.plot(times[:li],np.transpose(solute_alk[1,:li]), **kwargs_a)
plt.plot(times[:li],np.transpose(solute_alk[2,:li]), **kwargs_b)
plt.title('alk', fontsize=the_fs)


ax1=fig.add_subplot(2,4,3)
#plt.plot(times[:li],np.transpose(solute_c_mean[:li]), c=col[1], linestyle="-", lw=2, label='dic')
plt.plot(times[:li],np.transpose(solute_c[0,:li]), **kwargs_s)
#plt.plot(times[:li],np.transpose(solute_c[1,:li]), **kwargs_a)
#plt.plot(times[:li],np.transpose(solute_c[2,:li]), **kwargs_b)
plt.title('C', fontsize=the_fs)


ax1=fig.add_subplot(2,4,4)
#plt.plot(times[:li],np.transpose(solute_ca_mean[:li]), c=col[0], linestyle="-", lw=2, label='ca')
plt.plot(times[:li],np.transpose(solute_ca[0,:li]), **kwargs_s)
plt.plot(times[:li],np.transpose(solute_ca[1,:li]), **kwargs_a)
plt.plot(times[:li],np.transpose(solute_ca[2,:li]), **kwargs_b)
plt.title('Ca', fontsize=the_fs)


ax1=fig.add_subplot(2,4,5)
#plt.plot(times[:li],np.transpose(solute_mg_mean[:li]), **kwargs_s)
plt.plot(times[:li],np.transpose(solute_mg[0,:li]), **kwargs_s)
plt.plot(times[:li],np.transpose(solute_mg[1,:li]), **kwargs_a)
plt.plot(times[:li],np.transpose(solute_mg[2,:li]), **kwargs_b)
plt.title('Mg', fontsize=the_fs)



ax1=fig.add_subplot(2,4,6)
#plt.plot(times[:li],np.transpose(solute_k_mean[:li]), c=col[0], linestyle="-", lw=2, label='k')
plt.plot(times[:li],np.transpose(solute_k[0,:li]), **kwargs_s)
plt.plot(times[:li],np.transpose(solute_k[1,:li]), **kwargs_a)
plt.plot(times[:li],np.transpose(solute_k[2,:li]), **kwargs_b)
plt.title('K', fontsize=the_fs)



ax1=fig.add_subplot(2,4,7)
#plt.plot(times[:li],np.transpose(solute_si_mean[:li]), c=col[0], linestyle="-", lw=2, label='si')
plt.plot(times[:li],np.transpose(solute_si[0,:li]), **kwargs_s)
plt.plot(times[:li],np.transpose(solute_si[1,:li]), **kwargs_a)
plt.plot(times[:li],np.transpose(solute_si[2,:li]), **kwargs_b)
plt.title('Si', fontsize=the_fs)



ax1=fig.add_subplot(2,4,8)
#plt.plot(times[:li],np.transpose(solute_al_mean[:li]), c=col[0], linestyle="-", lw=2, label='al')
plt.plot(times[:li],np.transpose(solute_al[0,:li]), **kwargs_s)
plt.plot(times[:li],np.transpose(solute_al[1,:li]), **kwargs_a)
plt.plot(times[:li],np.transpose(solute_al[2,:li]), **kwargs_b)
plt.title('Al', fontsize=the_fs)



plt.subplots_adjust( wspace=0.5 , hspace=0.2 )
plt.savefig(outpath+'mat_solute_init'+'.png',bbox_inches='tight')
