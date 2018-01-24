# plot_box_q.py

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

tp = 1000
n_box = 3
minNum = 41

#todo: path here
prefix = "q_group_e/"
# path_label = prefix[3:7]
# print path_label

sample_outpath = "../output/revival/winter_basalt_box/"+prefix+"q_1.0_diff_10.00/"
outpath = "../output/revival/winter_basalt_box/"



#todo: PARAMS
param_strings = ['1.0', '2.0', '3.0', '4.0']
param_nums = [1.0, 2.0, 3.0, 4.0]
param_nums_time = [(3.14e7)*(0.006/1000.0)/((1.0e-9)*1.0), (3.14e7)*(0.006/1000.0)/((1.0e-9)*2.0), (3.14e7)*(0.006/1000.0)/((1.0e-9)*3.0), (3.14e7)*(0.006/1000.0)/((1.0e-9)*4.0)]

# diff_strings = ['10.00', '10.25', '10.50', '10.75', '11.00', '11.25', '11.50', '11.75', '12.00']
# diff_nums = [10.00, 10.25, 10.50, 10.75, 11.00, 11.25, 11.50, 11.75, 12.00]

diff_strings = ['10.00', '10.25', '10.50', '10.75', '11.00', '11.25', '11.50', '11.75']
diff_nums = [10.00, 10.25, 10.50, 10.75, 11.00, 11.25, 11.50, 11.75]



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
sec_full = np.zeros([3, minNum+1,tp,len(param_strings),len(diff_strings)])
sec = np.zeros([minNum+1,tp,len(param_strings),len(diff_strings)])
sec_d = np.zeros([minNum+1,tp,len(param_strings),len(diff_strings)])
sec_a = np.zeros([minNum+1,tp,len(param_strings),len(diff_strings)])
sec_b = np.zeros([minNum+1,tp,len(param_strings),len(diff_strings)])

dsec = np.zeros([minNum+1,tp,len(param_strings),len(diff_strings)])
dsec_d = np.zeros([minNum+1,tp,len(param_strings),len(diff_strings)])
dsec_a = np.zeros([minNum+1,tp,len(param_strings),len(diff_strings)])
dsec_b = np.zeros([minNum+1,tp,len(param_strings),len(diff_strings)])

pri_full = np.zeros([3,tp,len(param_strings),len(diff_strings)])
pri = np.zeros([tp,len(param_strings),len(diff_strings)])
pri_d = np.zeros([tp,len(param_strings),len(diff_strings)])
pri_a = np.zeros([tp,len(param_strings),len(diff_strings)])
pri_b = np.zeros([tp,len(param_strings),len(diff_strings)])

dpri = np.zeros([tp,len(param_strings),len(diff_strings)])
dpri_d = np.zeros([tp,len(param_strings),len(diff_strings)])
dpri_a = np.zeros([tp,len(param_strings),len(diff_strings)])
dpri_b = np.zeros([tp,len(param_strings),len(diff_strings)])

sol_full = np.zeros([3, 15,tp,len(param_strings),len(diff_strings)])
sol = np.zeros([15,tp,len(param_strings),len(diff_strings)])
sol_d = np.zeros([15,tp,len(param_strings),len(diff_strings)])
sol_a = np.zeros([15,tp,len(param_strings),len(diff_strings)])
sol_b = np.zeros([15,tp,len(param_strings),len(diff_strings)])

alk_in = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_out = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_flux = np.zeros([tp,len(param_strings),len(diff_strings)])

alk_in_d = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_out_d = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_flux_d = np.zeros([tp,len(param_strings),len(diff_strings)])

alk_in_a = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_out_a = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_flux_a = np.zeros([tp,len(param_strings),len(diff_strings)])

alk_in_b = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_out_b = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_flux_b = np.zeros([tp,len(param_strings),len(diff_strings)])

#todo: 2d arrays

value_dpri_mean = np.zeros([len(param_strings),len(diff_strings)])
value_dpri_mean_d = np.zeros([len(param_strings),len(diff_strings)])
value_dpri_mean_a = np.zeros([len(param_strings),len(diff_strings)])
value_dpri_mean_b = np.zeros([len(param_strings),len(diff_strings)])


value_sec_bin = np.zeros([minNum+1,len(param_strings),len(diff_strings)])
value_sec_bin_d = np.zeros([minNum+1,len(param_strings),len(diff_strings)])
value_sec_bin_a = np.zeros([minNum+1,len(param_strings),len(diff_strings)])
value_sec_bin_b = np.zeros([minNum+1,len(param_strings),len(diff_strings)])


value_sec = np.zeros([minNum+1,len(param_strings),len(diff_strings)])
value_sec_d = np.zeros([minNum+1,len(param_strings),len(diff_strings)])
value_sec_a = np.zeros([minNum+1,len(param_strings),len(diff_strings)])
value_sec_b = np.zeros([minNum+1,len(param_strings),len(diff_strings)])


value_dsec = np.zeros([minNum+1,len(param_strings),len(diff_strings)])
value_dsec_d = np.zeros([minNum+1,len(param_strings),len(diff_strings)])
value_dsec_a = np.zeros([minNum+1,len(param_strings),len(diff_strings)])
value_dsec_b = np.zeros([minNum+1,len(param_strings),len(diff_strings)])


value_alk_mean = np.zeros([len(param_strings),len(diff_strings)])
value_alk_mean_d = np.zeros([len(param_strings),len(diff_strings)])
value_alk_mean_a = np.zeros([len(param_strings),len(diff_strings)])
value_alk_mean_b = np.zeros([len(param_strings),len(diff_strings)])



value_sol = np.zeros([15,len(param_strings),len(diff_strings)])
value_sol_d = np.zeros([15,len(param_strings),len(diff_strings)])
value_sol_a = np.zeros([15,len(param_strings),len(diff_strings)])
value_sol_b = np.zeros([15,len(param_strings),len(diff_strings)])


value_dsol = np.zeros([15,len(param_strings),len(diff_strings)])
value_dsol_d = np.zeros([15,len(param_strings),len(diff_strings)])
value_dsol_a = np.zeros([15,len(param_strings),len(diff_strings)])
value_dsol_b = np.zeros([15,len(param_strings),len(diff_strings)])


any_min = []
for ii in range(len(param_strings)):
    for iii in range(len(diff_strings)):

        ii_path = "../output/revival/winter_basalt_box/"+prefix+"q_"+param_strings[ii]+"_diff_"+diff_strings[iii]+"/"
        print ii_path


        #todo: load in data
        for j in range(1,minNum):
            if os.path.isfile(ii_path + 'z_secondary_mat' + str(j) + '.txt'):
                if not np.any(any_min == j):
                    any_min = np.append(any_min,j)
                #print j , secondary[j] ,
                sec_full[:,j,:,ii,iii] = np.transpose(np.loadtxt(ii_path + 'z_secondary_mat' + str(j) + '.txt'))
                sec[j,:,ii,iii] = sec_full[0,j,:,ii,iii]
                sec_d[j,:,ii,iii] = sec_full[1,j,:,ii,iii] + sec_full[2,j,:,ii,iii]
                sec_a[j,:,ii,iii] = sec_full[1,j,:,ii,iii]
                sec_b[j,:,ii,iii] = sec_full[2,j,:,ii,iii]
        print param_strings[ii], any_min
        pri_full[:,:,ii,iii] = np.transpose(np.loadtxt(ii_path + 'z_primary_mat5.txt'))
        pri[:,ii,iii] = pri_full[0,:,ii,iii]
        pri_d[:,ii,iii] = pri_full[1,:,ii,iii] + pri_full[2,:,ii,iii]
        pri_a[:,ii,iii] = pri_full[1,:,ii,iii]
        pri_b[:,ii,iii] = pri_full[2,:,ii,iii]

        dpri[:-1,ii,iii] = pri[1:,ii,iii] - pri[:-1,ii,iii]
        dpri_d[:-1,ii,iii] = pri_d[1:,ii,iii] - pri_d[:-1,ii,iii]
        dpri_a[:-1,ii,iii] = pri_a[1:,ii,iii] - pri_a[:-1,ii,iii]
        dpri_b[:-1,ii,iii] = pri_b[1:,ii,iii] - pri_b[:-1,ii,iii]

        dsol_start = 100

        # import ph
        this_sol = 1
        sol_full[:,this_sol,:,ii,iii] = np.transpose(np.loadtxt(ii_path + 'z_solute_ph.txt'))
        sol[this_sol,:,ii,iii] = sol_full[0,this_sol,:,ii,iii]
        sol_a[this_sol,:,ii,iii] = sol_full[1,this_sol,:,ii,iii]
        sol_b[this_sol,:,ii,iii] = sol_full[2,this_sol,:,ii,iii]

        # import alk
        this_sol = 2
        sol_full[:,this_sol,:,ii,iii] = np.transpose(np.loadtxt(ii_path + 'z_solute_alk.txt'))
        sol[this_sol,:,ii,iii] = sol_full[0,this_sol,:,ii,iii]
        sol_a[this_sol,:,ii,iii] = sol_full[1,this_sol,:,ii,iii]
        sol_b[this_sol,:,ii,iii] = sol_full[2,this_sol,:,ii,iii]
        sol_d[this_sol,:,ii,iii] = (sol_a[this_sol,:,ii,iii] + sol_b[this_sol,:,ii,iii])/2.0

        value_dsol[this_sol,ii,iii] = np.mean(sol[this_sol,dsol_start:,ii,iii]-sol[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_a[this_sol,ii,iii] = np.mean(sol_a[this_sol,dsol_start:,ii,iii]-sol_a[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_b[this_sol,ii,iii] = np.mean(sol_b[this_sol,dsol_start:,ii,iii]-sol_b[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_d[this_sol,ii,iii] = np.mean(sol_d[this_sol,dsol_start:,ii,iii]-sol_d[this_sol,dsol_start-1:-1,ii,iii])

        # import sol_w
        this_sol = 3
        sol_full[:,this_sol,:,ii,iii] = np.transpose(np.loadtxt(ii_path + 'z_solute_w.txt'))
        sol[this_sol,:,ii,iii] = sol_full[0,this_sol,:,ii,iii]
        sol_a[this_sol,:,ii,iii] = sol_full[1,this_sol,:,ii,iii]
        sol_b[this_sol,:,ii,iii] = sol_full[2,this_sol,:,ii,iii]
        sol_d[this_sol,:,ii,iii] = (sol_a[this_sol,:,ii,iii] + sol_b[this_sol,:,ii,iii])/2.0

        # import c
        this_sol = 4
        sol_full[:,this_sol,:,ii,iii] = np.transpose(np.loadtxt(ii_path + 'z_solute_c.txt'))
        sol[this_sol,:,ii,iii] = sol_full[0,this_sol,:,ii,iii]
        sol_a[this_sol,:,ii,iii] = sol_full[1,this_sol,:,ii,iii]
        sol_b[this_sol,:,ii,iii] = sol_full[2,this_sol,:,ii,iii]
        sol_d[this_sol,:,ii,iii] = (sol_a[this_sol,:,ii,iii] + sol_b[this_sol,:,ii,iii])/2.0

        value_dsol[this_sol,ii,iii] = np.mean(sol[this_sol,dsol_start:,ii,iii]-sol[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_a[this_sol,ii,iii] = np.mean(sol_a[this_sol,dsol_start:,ii,iii]-sol_a[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_b[this_sol,ii,iii] = np.mean(sol_b[this_sol,dsol_start:,ii,iii]-sol_b[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_d[this_sol,ii,iii] = np.mean(sol_d[this_sol,dsol_start:,ii,iii]-sol_d[this_sol,dsol_start-1:-1,ii,iii])

        # import Ca
        this_sol = 5
        sol_full[:,this_sol,:,ii,iii] = np.transpose(np.loadtxt(ii_path + 'z_solute_ca.txt'))
        sol[this_sol,:,ii,iii] = sol_full[0,this_sol,:,ii,iii]
        sol_a[this_sol,:,ii,iii] = sol_full[1,this_sol,:,ii,iii]
        sol_b[this_sol,:,ii,iii] = sol_full[2,this_sol,:,ii,iii]
        sol_d[this_sol,:,ii,iii] = (sol_a[this_sol,:,ii,iii] + sol_b[this_sol,:,ii,iii])/2.0

        value_dsol[this_sol,ii,iii] = np.mean(sol[this_sol,dsol_start:,ii,iii]-sol[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_a[this_sol,ii,iii] = np.mean(sol_a[this_sol,dsol_start:,ii,iii]-sol_a[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_b[this_sol,ii,iii] = np.mean(sol_b[this_sol,dsol_start:,ii,iii]-sol_b[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_d[this_sol,ii,iii] = np.mean(sol_d[this_sol,dsol_start:,ii,iii]-sol_d[this_sol,dsol_start-1:-1,ii,iii])

        # import Mg
        this_sol = 6
        sol_full[:,this_sol,:,ii,iii] = np.transpose(np.loadtxt(ii_path + 'z_solute_mg.txt'))
        sol[this_sol,:,ii,iii] = sol_full[0,this_sol,:,ii,iii]
        sol_a[this_sol,:,ii,iii] = sol_full[1,this_sol,:,ii,iii]
        sol_b[this_sol,:,ii,iii] = sol_full[2,this_sol,:,ii,iii]
        sol_d[this_sol,:,ii,iii] = (sol_a[this_sol,:,ii,iii] + sol_b[this_sol,:,ii,iii])/2.0

        value_dsol[this_sol,ii,iii] = np.mean(sol[this_sol,dsol_start:,ii,iii]-sol[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_a[this_sol,ii,iii] = np.mean(sol_a[this_sol,dsol_start:,ii,iii]-sol_a[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_b[this_sol,ii,iii] = np.mean(sol_b[this_sol,dsol_start:,ii,iii]-sol_b[this_sol,dsol_start-1:-1,ii,iii])
        value_dsol_d[this_sol,ii,iii] = np.mean(sol_d[this_sol,dsol_start:,ii,iii]-sol_d[this_sol,dsol_start-1:-1,ii,iii])


        #todo: alk in/out
        alk_in[1:,ii,iii] = sol[3,1:,ii,iii] * (1.57e11)/((param_nums_time[ii])) * .00243
        alk_out[1:,ii,iii] = sol[3,1:,ii,iii] * (1.57e11)/((param_nums_time[ii])) * sol[2,1:,ii,iii] + 2.0*(sec[9,1:,ii,iii] - sec[9,:-1,ii,iii])
        alk_flux[:,ii,iii] = alk_out[:,ii,iii] - alk_in[:,ii,iii]

        alk_in_d[1:,ii,iii] = sol_d[3,1:,ii,iii] * 2.0*(1.57e11)/((param_nums_time[ii])) * .00243
        alk_out_d[1:,ii,iii] = sol_d[3,1:,ii,iii] * 2.0*(1.57e11)/((param_nums_time[ii])) * sol_d[2,1:,ii,iii] + 2.0*(sec_d[9,1:,ii,iii] - sec_d[9,:-1,ii,iii])
        alk_flux_d[:,ii,iii] = alk_out_d[:,ii,iii] - alk_in_d[:,ii,iii]

        alk_in_a[1:,ii,iii] = sol_a[3,1:,ii,iii] * 2.0*(1.57e11)/((param_nums_time[ii])) * .00243
        alk_out_a[1:,ii,iii] = sol_a[3,1:,ii,iii] * 2.0*(1.57e11)/((param_nums_time[ii])) * sol_a[2,1:,ii,iii] + 2.0*(sec_a[9,1:,ii,iii] - sec_a[9,:-1,ii,iii])
        alk_flux_a[:,ii,iii] = alk_out_a[:,ii,iii] - alk_in_a[:,ii,iii]

        # alk_in_b[1:,ii,iii] = sol_b[3,1:,ii,iii] * (1.57e11)/((param_nums_time[ii])) * .00243
        # alk_out_b[1:,ii,iii] = sol_b[3,1:,ii,iii] * (1.57e11)/((param_nums_time[ii])) * sol_b[2,1:,ii,iii] + 2.0*(sec_b[9,1:,ii,iii] - sec_b[9,:-1,ii,iii])
        # alk_flux_b[:,ii,iii] = alk_out_b[:,ii,iii] - alk_in_b[:,ii,iii]

        alk_in_b[1:,ii,iii] = sol_b[3,1:,ii,iii] * (1.57e11)/(10.0**(diff_nums[ii])) * .00243
        alk_out_b[1:,ii,iii] = sol_b[3,1:,ii,iii] * (1.57e11)/(10.0**(diff_nums[ii])) * sol_b[2,1:,ii,iii] + 2.0*(sec_b[9,1:,ii,iii] - sec_b[9,:-1,ii,iii])
        alk_flux_b[:,ii,iii] = alk_out_b[:,ii,iii] - alk_in_b[:,ii,iii]

        #alk_flux_d[:,ii,iii] = alk_flux_a[:,ii,iii] + alk_flux_b[:,ii,iii]


        value_alk_mean[ii,iii] = np.mean(alk_flux[:,ii,iii])
        value_alk_mean_d[ii,iii] = np.mean(alk_flux_d[:,ii,iii])
        value_alk_mean_a[ii,iii] = np.mean(alk_flux_a[:,ii,iii])
        value_alk_mean_b[ii,iii] = np.mean(alk_flux_b[:,ii,iii])


        print param_nums[ii]
        #print (sec[9,1:,ii] - sec[9,:-1,ii])
        # print alk_out[:,ii]- alk_in[:,ii]
        #print 10**(param_nums[ii])
        print " "

        value_dpri_mean[ii,iii] = np.mean(dpri[:,ii,iii])
        value_dpri_mean_d[ii,iii] = np.mean(dpri_d[:,ii,iii])
        value_dpri_mean_a[ii,iii] = np.mean(dpri_a[:,ii,iii])
        value_dpri_mean_b[ii,iii] = np.mean(dpri_b[:,ii,iii])



#todo: fill value_sec and value_sec_bin
for ii in range(len(param_strings)):
    for iii in range(len(diff_strings)):

        for j in range(len(any_min)):
            if np.max(sec[any_min[j],:,ii,iii]) > 0.0:
                value_sec_bin[any_min[j],ii,iii] = 1.0

            if np.max(sec_d[any_min[j],:,ii,iii]) > 0.0:
                value_sec_bin_d[any_min[j],ii,iii] = 1.0

            if np.max(sec_a[any_min[j],:,ii,iii]) > 0.0:
                value_sec_bin_a[any_min[j],ii,iii] = 1.0

            if np.max(sec_b[any_min[j],:,ii,iii]) > 0.0:
                value_sec_bin_b[any_min[j],ii,iii] = 1.0

            # fill value_sec with amount at last timestep, not mean
            value_sec[any_min[j],ii,iii] = sec[any_min[j],-1,ii,iii]
            value_sec_d[any_min[j],ii,iii] = sec_d[any_min[j],-1,ii,iii]
            value_sec_a[any_min[j],ii,iii] = sec_a[any_min[j],-1,ii,iii]
            value_sec_b[any_min[j],ii,iii] = sec_b[any_min[j],-1,ii,iii]



bar_bins = 4
bar_shrink = 0.9
xlabelpad = -2
clabelpad = 0















#hack: pri 2d plot
# primary slope?
fig=plt.figure(figsize=(8.0,8.0))



ax=fig.add_subplot(2, 2, 1, frameon=True)

the_xticks = range(len(diff_strings))
for i in the_xticks:
    the_xticks[i] = the_xticks[i] + 0.5
print "the_xticks" , the_xticks
plt.xticks(the_xticks,diff_strings, fontsize=8)
the_yticks = range(len(param_strings))
for i in the_yticks:
    the_yticks[i] = the_yticks[i] + 0.5
print "the_yticks" , the_yticks
plt.yticks(the_yticks,param_strings, fontsize=8)

plt.xlabel('param_t_diff')
plt.ylabel('param_q')

this_plot = np.abs(value_dpri_mean)
plt.pcolor(this_plot)

# print "this plot 1"
# print this_plot
# print " "

cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
cbar.ax.set_xlabel('dpri mean',fontsize=10,labelpad=clabelpad)





ax=fig.add_subplot(2, 2, 2, frameon=True)

plt.xticks(the_xticks,diff_strings, fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = np.abs(value_dpri_mean_d)
plt.pcolor(this_plot)

# print "this plot 2"
# print this_plot

cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
cbar.ax.set_xlabel('dpri_d mean',fontsize=10,labelpad=clabelpad)




ax=fig.add_subplot(2, 2, 3, frameon=True)

plt.xticks(the_xticks,diff_strings, fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = np.abs(value_dpri_mean_a)
plt.pcolor(this_plot)

# print "this plot 3"
# print this_plot

cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
cbar.ax.set_xlabel('dpri_a mean',fontsize=10,labelpad=clabelpad)




ax=fig.add_subplot(2, 2, 4, frameon=True)

plt.xticks(the_xticks,diff_strings, fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = np.abs(value_dpri_mean_b)
plt.pcolor(this_plot)
#
# print "this plot 4"
# print this_plot

cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
cbar.ax.set_xlabel('dpri_b mean',fontsize=10,labelpad=clabelpad)


plt.savefig(outpath+prefix+"x_2d_pri.png",bbox_inches='tight')








title_fs = 10

#hack: sec binary 2d plots
fig=plt.figure(figsize=(19.0,8.0))


for j in range(len(any_min)):
    ax=fig.add_subplot(4, (len(any_min)+1), j+1, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    this_plot = np.abs(value_sec_bin[any_min[j],:,:])
    plt.pcolor(this_plot,vmin=0.0,vmax=1.0)

    # print secondary[any_min[j]]
    # print this_plot
    # print " "

    plt.title(secondary[any_min[j]],fontsize=title_fs)






    ax=fig.add_subplot(4, (len(any_min)+1), (len(any_min)+1) + j+1, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    this_plot = np.abs(value_sec_bin_d[any_min[j],:,:])
    plt.pcolor(this_plot,vmin=0.0,vmax=1.0)

    #plt.title(secondary[any_min[j]],fontsize=title_fs)




    ax=fig.add_subplot(4, (len(any_min)+1), 2*(len(any_min)+1) + j+1, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    this_plot = np.abs(value_sec_bin_a[any_min[j],:,:])
    plt.pcolor(this_plot,vmin=0.0,vmax=1.0)

    #plt.title(secondary[any_min[j]],fontsize=title_fs)




    ax=fig.add_subplot(4, (len(any_min)+1), 3*(len(any_min)+1) + j+1, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    this_plot = np.abs(value_sec_bin_b[any_min[j],:,:])
    plt.pcolor(this_plot,vmin=0.0,vmax=1.0)

    #plt.title(secondary[any_min[j]],fontsize=title_fs)

    # cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
    # cbar.ax.set_xlabel('dpri_d mean',fontsize=10,labelpad=clabelpad)



plt.savefig(outpath+prefix+"x_2d_sec_bin.png",bbox_inches='tight')













#hack: calcite 2d plot
fig=plt.figure(figsize=(12.0,9.0))
this_min = 9


min_all = np.min(value_sec[this_min,:,:])
if np.min(value_sec_d[this_min,:,:]) < min_all:
    min_all = np.min(value_sec_d[this_min,:,:])
if np.min(value_sec_a[this_min,:,:]) < min_all:
    min_all = np.min(value_sec_a[this_min,:,:])
if np.min(value_sec_b[this_min,:,:]) < min_all:
    min_all = np.min(value_sec_b[this_min,:,:])

max_all = np.max(value_sec[this_min,:,:])
if np.max(value_sec_d[this_min,:,:]) > max_all:
    max_all = np.max(value_sec_d[this_min,:,:])
if np.max(value_sec_a[this_min,:,:]) > max_all:
    max_all = np.max(value_sec_a[this_min,:,:])
if np.max(value_sec_b[this_min,:,:]) > max_all:
    max_all = np.max(value_sec_b[this_min,:,:])


ax=fig.add_subplot(3, 4, 1, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

plt.xlabel('param_t_diff')
plt.ylabel('param_q')

this_plot = value_sec[this_min,:,:]
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

# cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
# cbar.ax.tick_params(labelsize=8)
# cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
# cbar.ax.set_xlabel('ch_s CaCO3 - end',fontsize=10)

bbox = ax.get_position()
cax = fig.add_axes([bbox.xmin+0.25, bbox.ymin-0.05, bbox.width*1.5, bbox.height*0.06])
cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
cbar = plt.colorbar(pCol, cax = cax,orientation='horizontal')
cbar.ax.tick_params(labelsize=7)
plt.title('CaCO3 at end',fontsize=9)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")


ax=fig.add_subplot(3, 4, 2, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_sec_d[this_min,:,:]
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

# cbar = plt.colorbar(orientation='horizontal',shrink=1.5)
# cbar.ax.tick_params(labelsize=8)
# cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
# cbar.ax.set_xlabel('ch_d CaCO3 - end',fontsize=10)







ax=fig.add_subplot(3, 4, 3, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_sec_a[this_min,:,:]
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

# cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
# cbar.ax.tick_params(labelsize=8)
# cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
# cbar.ax.set_xlabel('ch_a CaCO3 - end',fontsize=10)





ax=fig.add_subplot(3, 4, 4, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_sec_b[this_min,:,:]
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

# cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
# cbar.ax.tick_params(labelsize=8)
# cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
# cbar.ax.set_xlabel('ch_b CaCO3 - end',fontsize=10)







plt.savefig(outpath+prefix+"x_2d_calcite.png",bbox_inches='tight')













#hack: dsol 2d
fig=plt.figure(figsize=(12.0,9.0))

sp1 = 4
sp2 = 4

this_min = 2


min_all = np.min(value_dsol[this_min,:,:])
if np.min(value_dsol_d[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_d[this_min,:,:])
if np.min(value_dsol_a[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_a[this_min,:,:])
if np.min(value_dsol_b[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_b[this_min,:,:])

max_all = np.max(value_dsol[this_min,:,:])
if np.max(value_dsol_d[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_d[this_min,:,:])
if np.max(value_dsol_a[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_a[this_min,:,:])
if np.max(value_dsol_b[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_b[this_min,:,:])


print "this_min " , this_min
print min_all
print max_all
print " "


ax=fig.add_subplot(sp1, sp2, 1, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

plt.xlabel('param_t_diff')
plt.ylabel('param_q')

this_plot = value_dsol[this_min,:,:]
pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

bbox = ax.get_position()
cax = fig.add_axes([bbox.xmin+0.25, bbox.ymin+0.00, bbox.width*1.5, bbox.height*0.06])
cbar = plt.colorbar(pCol, cax = cax,orientation='horizontal')
cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
cbar.ax.tick_params(labelsize=7)
plt.title('alk dsol',fontsize=9)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")


ax=fig.add_subplot(sp1, sp2, 2, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_d[this_min,:,:]
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 3, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_a[this_min,:,:]
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 4, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_b[this_min,:,:]
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)







this_min = 4


min_all = np.min(value_dsol[this_min,:,:])
if np.min(value_dsol_d[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_d[this_min,:,:])
if np.min(value_dsol_a[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_a[this_min,:,:])
if np.min(value_dsol_b[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_b[this_min,:,:])

max_all = np.max(value_dsol[this_min,:,:])
if np.max(value_dsol_d[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_d[this_min,:,:])
if np.max(value_dsol_a[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_a[this_min,:,:])
if np.max(value_dsol_b[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_b[this_min,:,:])


print "this_min " , this_min
print min_all
print max_all
print " "

ax=fig.add_subplot(sp1, sp2, 5, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

plt.xlabel('param_t_diff')
plt.ylabel('param_q')

this_plot = value_dsol[this_min,:,:]
pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

bbox = ax.get_position()
cax = fig.add_axes([bbox.xmin+0.25, bbox.ymin-0.025, bbox.width*1.5, bbox.height*0.06])
cbar = plt.colorbar(pCol, cax = cax,orientation='horizontal')
cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
cbar.ax.tick_params(labelsize=7)
plt.title('[c] dsol',fontsize=9)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")


ax=fig.add_subplot(sp1, sp2, 6, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_d[this_min,:,:]
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 7, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_a[this_min,:,:]
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 8, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_b[this_min,:,:]
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)








this_min = 5


min_all = np.min(value_dsol[this_min,:,:])
if np.min(value_dsol_d[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_d[this_min,:,:])
if np.min(value_dsol_a[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_a[this_min,:,:])
if np.min(value_dsol_b[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_b[this_min,:,:])

max_all = np.max(value_dsol[this_min,:,:])
if np.max(value_dsol_d[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_d[this_min,:,:])
if np.max(value_dsol_a[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_a[this_min,:,:])
if np.max(value_dsol_b[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_b[this_min,:,:])


print "this_min " , this_min
print min_all
print max_all
print " "


ax=fig.add_subplot(sp1, sp2, 9, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

plt.xlabel('param_t_diff')
plt.ylabel('param_q')

this_plot = value_dsol[this_min,:,:]
pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

bbox = ax.get_position()
cax = fig.add_axes([bbox.xmin+0.25, bbox.ymin-0.04, bbox.width*1.5, bbox.height*0.06])
cbar = plt.colorbar(pCol, cax = cax,orientation='horizontal')
cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
cbar.ax.tick_params(labelsize=7)
plt.title('[Ca] dsol',fontsize=9)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")


ax=fig.add_subplot(sp1, sp2, 10, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_d[this_min,:,:]
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 11, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_a[this_min,:,:]
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 12, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_b[this_min,:,:]
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)








this_min = 6


min_all = np.min(value_dsol[this_min,:,:])
if np.min(value_dsol_d[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_d[this_min,:,:])
if np.min(value_dsol_a[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_a[this_min,:,:])
if np.min(value_dsol_b[this_min,:,:]) < min_all:
    min_all = np.min(value_dsol_b[this_min,:,:])

max_all = np.max(value_dsol[this_min,:,:])
if np.max(value_dsol_d[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_d[this_min,:,:])
if np.max(value_dsol_a[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_a[this_min,:,:])
if np.max(value_dsol_b[this_min,:,:]) > max_all:
    max_all = np.max(value_dsol_b[this_min,:,:])


print "this_min " , this_min
print min_all
print max_all
print " "


ax=fig.add_subplot(sp1, sp2, 13, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

plt.xlabel('param_t_diff')
plt.ylabel('param_q')

this_plot = value_dsol[this_min,:,:]
pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

bbox = ax.get_position()
cax = fig.add_axes([bbox.xmin+0.25, bbox.ymin-0.05, bbox.width*1.5, bbox.height*0.06])
cbar = plt.colorbar(pCol, cax = cax,orientation='horizontal')
cbar.ax.tick_params(labelsize=7)
cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
plt.title('[Mg] dsol',fontsize=9)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")


ax=fig.add_subplot(sp1, sp2, 14, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_d[this_min,:,:]
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 15, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_a[this_min,:,:]
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 16, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_dsol_b[this_min,:,:]
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




plt.subplots_adjust(hspace=0.5)
plt.savefig(outpath+prefix+"x_2d_dsol.png",bbox_inches='tight')












#hack: alk 2d plot
fig=plt.figure(figsize=(8.0,8.0))

#value_alk_mean_d = value_alk_mean_d/2.0

min_s_d = np.min(value_alk_mean)
if np.min(value_alk_mean_d) < min_s_d:
    min_s_d = np.min(value_alk_mean_d)

max_s_d = np.max(value_alk_mean)
if np.max(value_alk_mean_d) > max_s_d:
    max_s_d = np.max(value_alk_mean_d)



ax=fig.add_subplot(2, 2, 1, frameon=True)

plt.xticks(the_xticks,diff_strings, fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

plt.xlabel('param_t_diff')
plt.ylabel('param_q')

this_plot = value_alk_mean
plt.pcolor(this_plot, vmin=min_s_d, vmax=max_s_d)


cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks([min_s_d,max_s_d,0.0])
# cbar.set_ticks([np.min(this_plot),np.max(this_plot),0.0])
#cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
cbar.ax.set_xlabel('alk mean',fontsize=10,labelpad=clabelpad)

print "value_alk_mean"
print value_alk_mean
print " "


ax=fig.add_subplot(2, 2, 2, frameon=True)

plt.xticks(the_xticks,diff_strings, fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_alk_mean_d
plt.pcolor(this_plot, vmin=min_s_d, vmax=max_s_d)

cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks([min_s_d,max_s_d,0.0])
#cbar.set_ticks([np.min(this_plot),np.max(this_plot),0.0])
#cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
cbar.ax.set_xlabel('alk_d mean',fontsize=10,labelpad=clabelpad)


print "value_alk_mean_d"
print value_alk_mean_d
print " "


ax=fig.add_subplot(2, 2, 3, frameon=True)

plt.xticks(the_xticks,diff_strings, fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_alk_mean_a
plt.pcolor(this_plot)

cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks([np.min(this_plot),np.max(this_plot),0.0])
#cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
cbar.ax.set_xlabel('alk_a mean',fontsize=10,labelpad=clabelpad)




ax=fig.add_subplot(2, 2, 4, frameon=True)

plt.xticks(the_xticks,diff_strings, fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_alk_mean_b
plt.pcolor(this_plot)

cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks([np.min(this_plot),np.max(this_plot),0.0])
#cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
cbar.ax.set_xlabel('alk_b mean',fontsize=10,labelpad=clabelpad)


plt.savefig(outpath+prefix+"x_2d_alk.png",bbox_inches='tight')


















#hack: plot pri groups

fig=plt.figure(figsize=(14.0,11.0))


for iii in range(len(diff_strings)):

    ax=fig.add_subplot(4, len(diff_strings), iii+1, frameon=True)
    for ii in range(len(param_strings)):
        plt.plot(pri[:,ii,iii], label=param_strings[ii], c=plot_col[ii])
    if iii==0:
        plt.legend(fontsize=8,loc='best',ncol=2,labelspacing=-0.1,columnspacing=-0.1)
    plt.title("t_diff="+diff_strings[iii]+"(ch_s)")


    ax=fig.add_subplot(4, len(diff_strings), iii+len(diff_strings)+1, frameon=True)
    for ii in range(len(param_strings)):
        plt.plot(pri_d[:,ii,iii], label=param_strings[ii], c=plot_col[ii])
    plt.title("t_diff="+diff_strings[iii]+"(ch_d)")


    ax=fig.add_subplot(4, len(diff_strings), iii+2*len(diff_strings)+1, frameon=True)
    for ii in range(len(param_strings)):
        plt.plot(pri_a[:,ii,iii], label=param_strings[ii], c=plot_col[ii])
    plt.title("t_diff="+diff_strings[iii]+"(ch_a)")

    ax=fig.add_subplot(4, len(diff_strings), iii+3*len(diff_strings)+1, frameon=True)
    for ii in range(len(param_strings)):
        plt.plot(pri_b[:,ii,iii], label=param_strings[ii], c=plot_col[ii])
    plt.title("t_diff="+diff_strings[iii]+"(ch_b)")

#plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.savefig(outpath+prefix+"pri_groups.png",bbox_inches='tight')
#plt.savefig(outpath+prefix+"zps_pri"+path_label+"_.eps",bbox_inches='tight')



#hack: plot sec
fig=plt.figure(figsize=(18.0,6))


for j in range(len(any_min)):
    ax=fig.add_subplot(3, 7, j+1, frameon=True)
    this_min = any_min[j]
    for ii in range(len(param_strings)):
        plt.plot(sec[this_min,:,ii], label=param_strings[ii], c=plot_col[ii])
    if j == len(any_min)-1:
        plt.legend(fontsize=8,bbox_to_anchor=(2.0, 1.0),ncol=2,labelspacing=-0.1,columnspacing=-0.1)
    plt.title(secondary[this_min])



plt.subplots_adjust(wspace=0.3, hspace=0.2)
plt.savefig(outpath+prefix+"sec.png",bbox_inches='tight')
#plt.savefig(outpath+prefix+"zps_sec"+path_label+"_.eps",bbox_inches='tight')




the_lw = 1.5





#hack: plot sols
# fig=plt.figure(figsize=(14.0,7.0))
#
#
# # FIRST ROW: sol ch_s
# ax=fig.add_subplot(3, 6, 1, frameon=True)
# this_min = 1
# for ii in range(len(param_strings)):
#     plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
# plt.legend(fontsize=8,bbox_to_anchor=(2.0, 1.48),ncol=3,labelspacing=0.1,columnspacing=0.1)
# plt.title('pH all')
#
#
# ax=fig.add_subplot(3, 6, 2, frameon=True)
# this_min = 2
# for ii in range(len(param_strings)):
#     plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
# plt.title('alk all')
#
#
# ax=fig.add_subplot(3, 6, 3, frameon=True)
# this_min = 4
# for ii in range(len(param_strings)):
#     plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
# plt.title('c all')
#
#
# ax=fig.add_subplot(3, 6, 4, frameon=True)
# this_min = 5
# for ii in range(len(param_strings)):
#     plt.plot(sol[this_min,:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
# plt.title('Ca all')
#
#
#
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# plt.savefig(outpath+prefix+"sol.png",bbox_inches='tight')
# #plt.savefig(outpath+prefix+"zps_sol"+path_label+"_.eps",bbox_inches='tight')








#hack: plot alk
fig=plt.figure(figsize=(10.0,3.0))

# FIRST ROW: alk x 3 for ch_s
ax=fig.add_subplot(1, 3, 1, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(alk_out[1:,ii,0] - alk_in[1:,ii,0], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.legend(fontsize=8,bbox_to_anchor=(1.2, 1.2),ncol=2,labelspacing=0.1,columnspacing=0.1)
plt.title('alk_out - alk_in solo')


ax=fig.add_subplot(1, 3, 2, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(alk_in[1:,ii,0], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('alk_in solo')


ax=fig.add_subplot(1, 3, 3, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(alk_out[1:,ii,0], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('alk_out solo')



# # SECOND ROW FAST
# ax=fig.add_subplot(3, 3, 4, frameon=True)
# for ii in range(len(fast_swi_strings)):
#     plt.plot(alk_out[1:,ii] - alk_in[1:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
# #plt.legend(fontsize=8,bbox_to_anchor=(1.0, 1.1),ncol=3,labelspacing=0.1,columnspacing=0.1)
# plt.title('alk_out - alk_in fast')
#
#
# ax=fig.add_subplot(3, 3, 5, frameon=True)
# for ii in range(len(fast_swi_strings)):
#     plt.plot(alk_in[1:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
# plt.title('alk_in all fast')
#
#
# ax=fig.add_subplot(3, 3, 6, frameon=True)
# for ii in range(len(fast_swi_strings)):
#     plt.plot(alk_out[1:,ii], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
# plt.title('alk_out all fast')


plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig(outpath+prefix+"x_alk.png",bbox_inches='tight')
#plt.savefig(outpath+prefix+"zps_x_alk"+path_label+"_.eps",bbox_inches='tight')
