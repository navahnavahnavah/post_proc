#plot_bb_linear.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 10

plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

# col = ['maroon', 'r', 'darkorange', 'gold', 'lawngreen', 'g', 'darkcyan', 'c', 'b', 'navy','purple', 'm', 'hotpink', 'gray', 'k', 'sienna', 'saddlebrown']

col = ['#6e0202', '#fc385b', '#ff7411', '#19a702', '#00520d', '#00ffc2', '#609ff2', '#20267c','#8f00ff', '#ec52ff', '#6e6e6e', '#000000', '#c6813a', '#7d4e22', '#ffff00', '#df9a00', '#812700', '#6b3f67', '#0f9995', '#4d4d4d', '#d9d9d9', '#e9acff']

plot_col = ['#940000', '#d26618', '#fcae00', '#acde03', '#36aa00', '#35b5aa', '#0740d2', '#7f05d4', '#b100de']

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

tp = 10000
n_box = 3
minNum = 41

#todo: path here
prefix = "swi_k88/mix_"
path_label = prefix[3:7]
print path_label

sample_outpath = "../output/revival/winter_basalt_box/"+prefix+"1e1100/"
outpath = "../output/revival/winter_basalt_box/"

# param_strings = ['1e11', '1e12', '1e13', '1e14']
# param_nums = [11, 12, 13, 14]

# param_strings = ['1e1100', '1e1125', '1e1150', '1e1175', '1e1250', '1e1250', '1e1300', '1e1350', '1e1400']
# param_nums = [11.0, 11.25, 11.50, 11.75, 12.5, 12.5, 13.0, 13.5, 14.0]

param_strings = ['1e1100', '1e1125', '1e1150', '1e1175', '1e1200']
param_nums = [11.0, 11.25, 11.50, 11.75, 12.0]

# fast_swi_strings = param_strings[0:4]
# print "fast_swi_strings: " , fast_swi_strings
# fast_swi_nums = param_nums[0:4]
#
# print " "
#
# slow_swi_strings = param_strings[4:]
# print "slow_swi_strings: " , slow_swi_strings
# slow_swi_nums = param_nums[4:]
#
# print " "

fast_slow = 3

fast_swi_strings = param_strings[0:fast_slow]
print "fast_swi_strings: " , fast_swi_strings
fast_swi_nums = param_nums[0:fast_slow]

print " "

slow_swi_strings = param_strings[fast_slow:]
print "slow_swi_strings: " , slow_swi_strings
slow_swi_nums = param_nums[fast_slow:]

print " "




#todo: big arrays go here
sec_full = np.zeros([3, minNum+1,tp,len(param_strings)])
sec = np.zeros([minNum+1,tp,len(param_strings)])
sec_d = np.zeros([minNum+1,tp,len(param_strings)])
sec_a = np.zeros([minNum+1,tp,len(param_strings)])
sec_b = np.zeros([minNum+1,tp,len(param_strings)])

dsec = np.zeros([minNum+1,tp,len(param_strings)])
dsec_d = np.zeros([minNum+1,tp,len(param_strings)])
dsec_a = np.zeros([minNum+1,tp,len(param_strings)])
dsec_b = np.zeros([minNum+1,tp,len(param_strings)])

pri_full = np.zeros([3,tp,len(param_strings)])
pri = np.zeros([tp,len(param_strings)])
pri_d = np.zeros([tp,len(param_strings)])
pri_a = np.zeros([tp,len(param_strings)])
pri_b = np.zeros([tp,len(param_strings)])

dpri = np.zeros([tp,len(param_strings)])
dpri_d = np.zeros([tp,len(param_strings)])
dpri_a = np.zeros([tp,len(param_strings)])
dpri_b = np.zeros([tp,len(param_strings)])

sol_full = np.zeros([3, 15,tp,len(param_strings)])
sol = np.zeros([15,tp,len(param_strings)])
sol_d = np.zeros([15,tp,len(param_strings)])
sol_a = np.zeros([15,tp,len(param_strings)])
sol_b = np.zeros([15,tp,len(param_strings)])

alk_in = np.zeros([tp,len(param_strings)])
alk_out = np.zeros([tp,len(param_strings)])


any_min = []
for ii in range(len(param_strings)):
    ii_path = "../output/revival/winter_basalt_box/"+prefix+param_strings[ii]+"/"
    print ii_path


    #todo: load in data
    for j in range(1,minNum):
        if os.path.isfile(ii_path + 'z_secondary_mat' + str(j) + '.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            #print j , secondary[j] ,
            sec_full[:,j,:,ii] = np.transpose(np.loadtxt(ii_path + 'z_secondary_mat' + str(j) + '.txt'))
            sec[j,:,ii] = sec_full[0,j,:,ii]
            sec_d[j,:,ii] = sec_full[1,j,:,ii] + sec_full[2,j,:,ii]
            sec_a[j,:,ii] = sec_full[1,j,:,ii]
            sec_b[j,:,ii] = sec_full[2,j,:,ii]
    print param_strings[ii], any_min
    pri_full[:,:,ii] = np.transpose(np.loadtxt(ii_path + 'z_primary_mat5.txt'))
    pri[:,ii] = pri_full[0,:,ii]
    pri_d[:,ii] = pri_full[1,:,ii] + pri_full[2,:,ii]
    pri_a[:,ii] = pri_full[1,:,ii]
    pri_b[:,ii] = pri_full[2,:,ii]

    # import ph
    this_sol = 1
    sol_full[:,this_sol,:,ii] = np.transpose(np.loadtxt(ii_path + 'z_solute_ph.txt'))
    sol[this_sol,:,ii] = sol_full[0,this_sol,:,ii]
    sol_a[this_sol,:,ii] = sol_full[1,this_sol,:,ii]
    sol_b[this_sol,:,ii] = sol_full[2,this_sol,:,ii]

    # import alk
    this_sol = 2
    sol_full[:,this_sol,:,ii] = np.transpose(np.loadtxt(ii_path + 'z_solute_alk.txt'))
    sol[this_sol,:,ii] = sol_full[0,this_sol,:,ii]
    sol_a[this_sol,:,ii] = sol_full[1,this_sol,:,ii]
    sol_b[this_sol,:,ii] = sol_full[2,this_sol,:,ii]

    # import sol_w
    this_sol = 3
    sol_full[:,this_sol,:,ii] = np.transpose(np.loadtxt(ii_path + 'z_solute_w.txt'))
    sol[this_sol,:,ii] = sol_full[0,this_sol,:,ii]
    sol_a[this_sol,:,ii] = sol_full[1,this_sol,:,ii]
    sol_b[this_sol,:,ii] = sol_full[2,this_sol,:,ii]

    # import c
    this_sol = 4
    sol_full[:,this_sol,:,ii] = np.transpose(np.loadtxt(ii_path + 'z_solute_c.txt'))
    sol[this_sol,:,ii] = sol_full[0,this_sol,:,ii]
    sol_a[this_sol,:,ii] = sol_full[1,this_sol,:,ii]
    sol_b[this_sol,:,ii] = sol_full[2,this_sol,:,ii]

    # import Ca
    this_sol = 5
    sol_full[:,this_sol,:,ii] = np.transpose(np.loadtxt(ii_path + 'z_solute_ca.txt'))
    sol[this_sol,:,ii] = sol_full[0,this_sol,:,ii]
    sol_a[this_sol,:,ii] = sol_full[1,this_sol,:,ii]
    sol_b[this_sol,:,ii] = sol_full[2,this_sol,:,ii]

    # import Ca
    this_sol = 6
    sol_full[:,this_sol,:,ii] = np.transpose(np.loadtxt(ii_path + 'z_solute_mg.txt'))
    sol[this_sol,:,ii] = sol_full[0,this_sol,:,ii]
    sol_a[this_sol,:,ii] = sol_full[1,this_sol,:,ii]
    sol_b[this_sol,:,ii] = sol_full[2,this_sol,:,ii]


    #todo: alk in/out
    alk_in[1:,ii] = sol[3,1:,ii] * (1.57e10)/(10**(param_nums[ii])) * .00243
    # print "alk_in" , param_nums[ii]
    # print alk_in[:,ii]

    alk_out[1:,ii] = sol[3,1:,ii] * (1.57e10)/(10**(param_nums[ii])) * sol[2,1:,ii] + 2.0*(sec[9,1:,ii] - sec[9,:-1,ii])

    # print "alk_out" , param_nums[ii]
    # print alk_out[:,ii]

    print param_nums[ii]
    #print (sec[9,1:,ii] - sec[9,:-1,ii])
    # print alk_out[:,ii]- alk_in[:,ii]
    #print 10**(param_nums[ii])
    print " "





#hack: plot sec
fig=plt.figure(figsize=(18.0,6))

# ax=fig.add_subplot(2, 7, 1, frameon=True)
# this_min = 2
# for ii in range(len(param_strings)):
#     plt.plot(sec[this_min,:,ii], label=param_strings[ii], c=plot_col[ii])
# plt.legend(fontsize=8,bbox_to_anchor=(1.1, 1.48),ncol=2,labelspacing=-0.1,columnspacing=-0.1)
# plt.title(secondary[this_min])

for j in range(len(any_min)):
    ax=fig.add_subplot(3, 7, j+1, frameon=True)
    this_min = any_min[j]
    for ii in range(len(param_strings)):
        plt.plot(sec[this_min,:,ii], label=param_strings[ii], c=plot_col[ii])
    if j == len(any_min)-1:
        plt.legend(fontsize=8,bbox_to_anchor=(2.0, 1.0),ncol=2,labelspacing=-0.1,columnspacing=-0.1)
    plt.title(secondary[this_min])



plt.subplots_adjust(wspace=0.3, hspace=0.2)
plt.savefig(outpath+prefix+"sec"+path_label+"_.png",bbox_inches='tight')
plt.savefig(outpath+prefix+"zps_sec"+path_label+"_.eps",bbox_inches='tight')






#hack: plot pri/sol_w
fig=plt.figure(figsize=(7.0,7.0))

ax=fig.add_subplot(2, 2, 1, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(pri[:,ii], label=param_strings[ii], c=plot_col[ii])
    plt.legend(fontsize=8,loc='center right',ncol=2,labelspacing=-0.1,columnspacing=-0.1)
plt.title('pri totals')


ax=fig.add_subplot(2, 2, 2, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(sol[3,:,ii], label=param_strings[ii], c=plot_col[ii])
plt.title('sol_w')

plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.savefig(outpath+prefix+"pri"+path_label+"_.png",bbox_inches='tight')
plt.savefig(outpath+prefix+"zps_pri"+path_label+"_.eps",bbox_inches='tight')



#hack: plot sols
fig=plt.figure(figsize=(14.0,7.0))

the_lw = 1.5

# sol ALL plots FIRST ROW
ax=fig.add_subplot(3, 6, 1, frameon=True)
this_min = 1
for ii in range(len(param_strings)):
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.legend(fontsize=8,bbox_to_anchor=(2.0, 1.48),ncol=3,labelspacing=0.1,columnspacing=0.1)
plt.title('pH all')


ax=fig.add_subplot(3, 6, 2, frameon=True)
this_min = 2
for ii in range(len(param_strings)):
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('alk all')


ax=fig.add_subplot(3, 6, 3, frameon=True)
this_min = 4
for ii in range(len(param_strings)):
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('c all')


ax=fig.add_subplot(3, 6, 4, frameon=True)
this_min = 5
for ii in range(len(param_strings)):
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('Ca all')



# sol FAST plots SECOND ROW
ax=fig.add_subplot(3, 6, 7, frameon=True)
this_min = 1
for ii in range(len(fast_swi_strings)):
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
#plt.legend(fontsize=8,loc='best',ncol=2,labelspacing=-0.1,columnspacing=-0.1)
plt.title('pH fast')


ax=fig.add_subplot(3, 6, 8, frameon=True)
this_min = 2
for ii in range(len(fast_swi_strings)):
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('alk fast')


ax=fig.add_subplot(3, 6, 9, frameon=True)
this_min = 4
for ii in range(len(fast_swi_strings)):
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('c fast')


ax=fig.add_subplot(3, 6, 10, frameon=True)
this_min = 5
for ii in range(len(fast_swi_strings)):
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('Ca fast')




# sol SLOW plots THIRD ROW
ax=fig.add_subplot(3, 6, 13, frameon=True)
this_min = 1
for ii in range(len(slow_swi_strings)):
    ii = ii + len(fast_swi_strings)
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
#plt.legend(fontsize=8,loc='best',ncol=2,labelspacing=-0.1,columnspacing=-0.1)
plt.title('pH slow')


ax=fig.add_subplot(3, 6, 14, frameon=True)
this_min = 2
for ii in range(len(slow_swi_strings)):
    ii = ii + len(fast_swi_strings) -0
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('alk slow')


ax=fig.add_subplot(3, 6, 15, frameon=True)
this_min = 4
for ii in range(len(slow_swi_strings)):
    ii = ii + len(fast_swi_strings) -0
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('c slow')


ax=fig.add_subplot(3, 6, 16, frameon=True)
this_min = 5
for ii in range(len(slow_swi_strings)):
    ii = ii + len(fast_swi_strings) -0
    plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('Ca slow')




plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig(outpath+prefix+"sol"+path_label+"_.png",bbox_inches='tight')
plt.savefig(outpath+prefix+"zps_sol"+path_label+"_.eps",bbox_inches='tight')



#hack: plot alk
fig=plt.figure(figsize=(10.0,9.0))

ax=fig.add_subplot(3, 3, 1, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(alk_out[1:,ii] - alk_in[1:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.legend(fontsize=8,bbox_to_anchor=(1.0, 1.2),ncol=3,labelspacing=0.1,columnspacing=0.1)
plt.title('alk_out - alk_in all')


ax=fig.add_subplot(3, 3, 2, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(alk_in[1:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('alk_in all')


ax=fig.add_subplot(3, 3, 3, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(alk_out[1:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('alk_out all')



# SECOND ROW FAST
ax=fig.add_subplot(3, 3, 4, frameon=True)
for ii in range(len(fast_swi_strings)):
    plt.plot(alk_out[1:,ii] - alk_in[1:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
#plt.legend(fontsize=8,bbox_to_anchor=(1.0, 1.1),ncol=3,labelspacing=0.1,columnspacing=0.1)
plt.title('alk_out - alk_in fast')


ax=fig.add_subplot(3, 3, 5, frameon=True)
for ii in range(len(fast_swi_strings)):
    plt.plot(alk_in[1:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('alk_in all fast')


ax=fig.add_subplot(3, 3, 6, frameon=True)
for ii in range(len(fast_swi_strings)):
    plt.plot(alk_out[1:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('alk_out all fast')


plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig(outpath+prefix+"x_alk"+path_label+"_.png",bbox_inches='tight')
plt.savefig(outpath+prefix+"zps_x_alk"+path_label+"_.eps",bbox_inches='tight')
