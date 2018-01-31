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
prefix = "y_group_b/"
# path_label = prefix[3:7]
# print path_label

sample_outpath = "../output/revival/winter_basalt_box/"+prefix+"q_1.0_diff_10.00/"
outpath = "../output/revival/winter_basalt_box/"



#todo: PARAMS
# param_strings = ['1.0', '2.0', '3.0', '4.0']
# param_nums = [1.0, 2.0, 3.0, 4.0]
# param_nums_time = [(3.14e7)*(0.006/1000.0)/((1.0e-9)*1.0), (3.14e7)*(0.006/1000.0)/((1.0e-9)*2.0), (3.14e7)*(0.006/1000.0)/((1.0e-9)*3.0), (3.14e7)*(0.006/1000.0)/((1.0e-9)*4.0)]

param_strings = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5']
param_nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
param_nums_time = np.zeros(len(param_nums))
for iii in range(len(param_nums)):
    #print param_nums[iii]
    param_nums_time[iii] = (3.14e7)*(0.006/1000.0)/((1.0e-9)*param_nums[iii])
    print param_nums[iii], param_nums_time[iii]
#param_nums_time = [(3.14e7)*(0.006/1000.0)/((1.0e-9)*0.5), (3.14e7)*(0.006/1000.0)/((1.0e-9)*1.0), (3.14e7)*(0.006/1000.0)/((1.0e-9)*1.5), (3.14e7)*(0.006/1000.0)/((1.0e-9)*2.0), (3.14e7)*(0.006/1000.0)/((1.0e-9)*2.5), (3.14e7)*(0.006/1000.0)/((1.0e-9)*3.0), (3.14e7)*(0.006/1000.0)/((1.0e-9)*3.5),  (3.14e7)*(0.006/1000.0)/((1.0e-9)*4.0), (3.14e7)*(0.006/1000.0)/((1.0e-9)*4.5)]

# diff_strings = ['10.00', '10.25', '10.50', '10.75', '11.00', '11.25', '11.50', '11.75', '12.00']
# diff_nums = [10.00, 10.25, 10.50, 10.75, 11.00, 11.25, 11.50, 11.75, 12.00]











# diff_strings = ['10.00', '10.25', '10.50', '10.75', '11.00', '11.25', '11.50']
# diff_nums = [10.00, 10.25, 10.50, 10.75, 11.00, 11.25, 11.50]

# diff_strings = ['2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50', '3.75', '4.00', '4.25', '4.50']
# diff_nums = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5]

diff_strings = ['2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50', '3.75', '4.00']
diff_nums = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]



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


## alk vol

alk_vol_in = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_vol_out = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_vol_flux = np.zeros([tp,len(param_strings),len(diff_strings)])

alk_vol_in_d = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_vol_out_d = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_vol_flux_d = np.zeros([tp,len(param_strings),len(diff_strings)])

alk_vol_in_a = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_vol_out_a = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_vol_flux_a = np.zeros([tp,len(param_strings),len(diff_strings)])

alk_vol_in_b = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_vol_out_b = np.zeros([tp,len(param_strings),len(diff_strings)])
alk_vol_flux_b = np.zeros([tp,len(param_strings),len(diff_strings)])


# alt_vol% and FeO/FeOt arrays here


alt_vol = np.zeros([tp,len(param_strings),len(diff_strings)])
alt_vol_a = np.zeros([tp,len(param_strings),len(diff_strings)])
alt_vol_b = np.zeros([tp,len(param_strings),len(diff_strings)])
alt_vol_d = np.zeros([tp,len(param_strings),len(diff_strings)])


alt_fe = np.zeros([tp,len(param_strings),len(diff_strings)])
alt_fe_a = np.zeros([tp,len(param_strings),len(diff_strings)])
alt_fe_b = np.zeros([tp,len(param_strings),len(diff_strings)])
alt_fe_d = np.zeros([tp,len(param_strings),len(diff_strings)])


value_alt_vol_mean = np.zeros([len(param_strings),len(diff_strings)])
value_alt_vol_mean_a = np.zeros([len(param_strings),len(diff_strings)])
value_alt_vol_mean_b = np.zeros([len(param_strings),len(diff_strings)])
value_alt_vol_mean_d = np.zeros([len(param_strings),len(diff_strings)])


value_alt_fe_mean = np.zeros([len(param_strings),len(diff_strings)])
value_alt_fe_mean_a = np.zeros([len(param_strings),len(diff_strings)])
value_alt_fe_mean_b = np.zeros([len(param_strings),len(diff_strings)])
value_alt_fe_mean_d = np.zeros([len(param_strings),len(diff_strings)])


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


value_alk_vol_mean = np.zeros([len(param_strings),len(diff_strings)])
value_alk_vol_mean_d = np.zeros([len(param_strings),len(diff_strings)])
value_alk_vol_mean_a = np.zeros([len(param_strings),len(diff_strings)])
value_alk_vol_mean_b = np.zeros([len(param_strings),len(diff_strings)])



value_sol = np.zeros([15,len(param_strings),len(diff_strings)])
value_sol_d = np.zeros([15,len(param_strings),len(diff_strings)])
value_sol_a = np.zeros([15,len(param_strings),len(diff_strings)])
value_sol_b = np.zeros([15,len(param_strings),len(diff_strings)])


value_dsol = np.zeros([15,len(param_strings),len(diff_strings)])
value_dsol_d = np.zeros([15,len(param_strings),len(diff_strings)])
value_dsol_a = np.zeros([15,len(param_strings),len(diff_strings)])
value_dsol_b = np.zeros([15,len(param_strings),len(diff_strings)])



#todo: net uptake arrays
x_elements = np.zeros([tp,15,len(param_strings),len(diff_strings)])
x_elements_d = np.zeros([tp,15,len(param_strings),len(diff_strings)])
x_elements_a = np.zeros([tp,15,len(param_strings),len(diff_strings)])
x_elements_b = np.zeros([tp,15,len(param_strings),len(diff_strings)])

x_pri_elements = np.zeros([tp,15,len(param_strings),len(diff_strings)])
x_pri_elements_d = np.zeros([tp,15,len(param_strings),len(diff_strings)])
x_pri_elements_a = np.zeros([tp,15,len(param_strings),len(diff_strings)])
x_pri_elements_b = np.zeros([tp,15,len(param_strings),len(diff_strings)])

elements_sec = np.zeros([minNum+1,15])
elements_pri = np.zeros([1,15])

elements_pri[0,5] = 0.2151 # Ca
elements_pri[0,6] = 0.178 # Mg
elements_pri[0,7] = 0.086 # Na
elements_pri[0,8] = 0.004 # K
elements_pri[0,9] = 0.2128 # Fe
elements_pri[0,10] = 0.0 # S
elements_pri[0,11] =  0.85 # Si
elements_pri[0,12] = 0.0 # Cl
elements_pri[0,13] =  0.28 # Al

# 2 saponite_mg
elements_sec[2,5] = 0.0 # Ca
elements_sec[2,6] = 3.165 # Mg
elements_sec[2,7] = 0.0 # Na
elements_sec[2,8] = 0.0 # K
elements_sec[2,9] = 0.0 # Fe
elements_sec[2,10] = 0.0 # S
elements_sec[2,11] = 3.67 # Si
elements_sec[2,12] = 0.0 # Cl
elements_sec[2,13] = 0.33 # Al

# 5 pyrite
elements_sec[5,5] = 0.0 # Ca
elements_sec[5,6] = 0.0 # Mg
elements_sec[5,7] = 0.0 # Na
elements_sec[5,8] = 0.0 # K
elements_sec[5,9] = 1.0 # Fe
elements_sec[5,10] = 2.0 # S
elements_sec[5,11] = 0.0 # Si
elements_sec[5,12] = 0.0 # Cl
elements_sec[5,13] = 0.0 # Al


# 9 calcite
elements_sec[9,5] = 1.0 # Ca
elements_sec[9,6] = 0.0 # Mg
elements_sec[9,7] = 0.0 # Na
elements_sec[9,8] = 0.0 # K
elements_sec[9,9] = 0.0 # Fe
elements_sec[9,10] = 0.0 # S
elements_sec[9,11] = 0.0 # Si
elements_sec[9,12] = 0.0 # Cl
elements_sec[9,13] = 0.0 # Al

# saponite_na
elements_sec[11,5] = 0.0 # Ca
elements_sec[11,6] = 3.0 # Mg
elements_sec[11,7] = 0.33 # Na
elements_sec[11,8] = 0.0 # K
elements_sec[11,9] = 0.0 # Fe
elements_sec[11,10] = 0.0 # S
elements_sec[11,11] = 3.67 # Si
elements_sec[11,12] = 0.0 # Cl
elements_sec[11,13] = 0.33 # Al

# 13 nont_mg
elements_sec[13,5] = 0.0 # Ca
elements_sec[13,6] = 0.165 # Mg
elements_sec[13,7] = 0.0 # Na
elements_sec[13,8] = 0.0 # K
elements_sec[13,9] = 2.0 # Fe
elements_sec[13,10] = 0.0 # S
elements_sec[13,11] = 3.67 # Si
elements_sec[13,12] = 0.0 # Cl
elements_sec[13,13] = 0.33 # Al

# 14 fe_celad
elements_sec[14,5] = 0.0 # Ca
elements_sec[14,6] = 0.0 # Mg
elements_sec[14,7] = 0.0 # Na
elements_sec[14,8] = 1.0 # K
elements_sec[14,9] = 1.0 # Fe
elements_sec[14,10] = 0.0 # S
elements_sec[14,11] = 4.0 # Si
elements_sec[14,12] = 0.0 # Cl
elements_sec[14,13] = 1.0 # Al

# 16 mesolite
elements_sec[16,5] = 0.657 # Ca
elements_sec[16,6] = 0.0 # Mg
elements_sec[16,7] = 0.676 # Na
elements_sec[16,8] = 0.0 # K
elements_sec[16,9] = 0.0 # Fe
elements_sec[16,10] = 0.0 # S
elements_sec[16,11] = 3.01 # Si
elements_sec[16,12] = 0.0 # Cl
elements_sec[16,13] = 1.99 # Al

# 17 hematite
elements_sec[17,5] = 0.0 # Ca
elements_sec[17,6] = 0.0 # Mg
elements_sec[17,7] = 0.0 # Na
elements_sec[17,8] = 0.0 # K
elements_sec[17,9] = 2.0 # Fe
elements_sec[17,10] = 0.0 # S
elements_sec[17,11] = 0.0 # Si
elements_sec[17,12] = 0.0 # Cl
elements_sec[17,13] = 0.0 # Al

# clinochlore14a
elements_sec[31,5] = 0.0 # Ca
elements_sec[31,6] = 5.0 # Mg
elements_sec[31,7] = 0.0 # Na
elements_sec[31,8] = 0.0 # K
elements_sec[31,9] = 0.0 # Fe
elements_sec[31,10] = 0.0 # S
elements_sec[31,11] = 3.0 # Si
elements_sec[31,12] = 0.0 # Cl
elements_sec[31,13] = 2.0 # Al

# saponite_ca
elements_sec[33,5] = 0.165 # Ca
elements_sec[33,6] = 3.0 # Mg
elements_sec[33,7] = 0.0 # Na
elements_sec[33,8] = 0.0 # K
elements_sec[33,9] = 0.0 # Fe
elements_sec[33,10] = 0.0 # S
elements_sec[33,11] = 3.67 # Si
elements_sec[33,12] = 0.0 # Cl
elements_sec[33,13] = 0.33 # Al


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

                dsec[j,1:,ii,iii] = sec[j,1:,ii,iii] - sec[j,:-1,ii,iii]
                dsec_d[j,1:,ii,iii] = sec_d[j,1:,ii,iii] - sec_d[j,:-1,ii,iii]
                dsec_a[j,1:,ii,iii] = sec_a[j,1:,ii,iii] - sec_a[j,:-1,ii,iii]
                dsec_b[j,1:,ii,iii] = sec_b[j,1:,ii,iii] - sec_b[j,:-1,ii,iii]
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

        alk_in_d[1:,ii,iii] = sol_a[3,1:,ii,iii] * 2.0*(1.57e11)/((param_nums_time[ii])) * .00243
        alk_out_d[1:,ii,iii] = sol_a[3,1:,ii,iii] * 2.0*(1.57e11)/((param_nums_time[ii])) * sol_a[2,1:,ii,iii] + 2.0*(sec_d[9,1:,ii,iii] - sec_d[9,:-1,ii,iii])
        alk_flux_d[:,ii,iii] = alk_out_d[:,ii,iii] - alk_in_d[:,ii,iii]

        alk_in_a[1:,ii,iii] = sol_a[3,1:,ii,iii] * 2.0*(1.57e11)/((param_nums_time[ii])) * .00243
        alk_out_a[1:,ii,iii] = sol_a[3,1:,ii,iii] * 2.0*(1.57e11)/((param_nums_time[ii])) * sol_a[2,1:,ii,iii] + 2.0*(sec_a[9,1:,ii,iii] - sec_a[9,:-1,ii,iii])
        alk_flux_a[:,ii,iii] = alk_out_a[:,ii,iii] - alk_in_a[:,ii,iii]

        # alk_in_b[1:,ii,iii] = sol_b[3,1:,ii,iii] * (1.57e11)/((param_nums_time[ii])) * .00243
        # alk_out_b[1:,ii,iii] = sol_b[3,1:,ii,iii] * (1.57e11)/((param_nums_time[ii])) * sol_b[2,1:,ii,iii] + 2.0*(sec_b[9,1:,ii,iii] - sec_b[9,:-1,ii,iii])
        # alk_flux_b[:,ii,iii] = alk_out_b[:,ii,iii] - alk_in_b[:,ii,iii]

        alk_in_b[1:,ii,iii] = sol_b[3,1:,ii,iii] * (1.57e11)/(10.0**(diff_nums[iii])) * .00243
        alk_out_b[1:,ii,iii] = sol_b[3,1:,ii,iii] * (1.57e11)/(10.0**(diff_nums[iii])) * sol_b[2,1:,ii,iii] + 2.0*(sec_b[9,1:,ii,iii] - sec_b[9,:-1,ii,iii])
        alk_flux_b[:,ii,iii] = alk_out_b[:,ii,iii] - alk_in_b[:,ii,iii]

        #alk_flux_d[:,ii,iii] = alk_flux_a[:,ii,iii] + alk_flux_b[:,ii,iii]


        alk_vol_in[1:,ii,iii] = 0.00243
        alk_vol_out[1:,ii,iii] = sol[2,1:,ii,iii] + 2.0*(sec[9,1:,ii,iii] - sec[9,:-1,ii,iii])/((1.57e11)/param_nums_time[ii])
        alk_vol_flux[:,ii,iii] = alk_vol_out[:,ii,iii] - alk_vol_in[:,ii,iii]


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


        #todo: net uptake data
        for j in range(len(any_min)):
                for jj in range(15):
                    x_elements[:,jj,ii,iii] = x_elements[:,jj,ii,iii] + elements_sec[any_min[j],jj]*dsec[any_min[j],:,ii,iii]
                    if j == 0:
                        x_pri_elements[:,jj,ii,iii] = elements_pri[0,jj]*dpri[:,ii,iii]


        #todo: alt_vol% data
        for j in range(len(any_min)):
            alt_vol[:,ii,iii] = alt_vol[:,ii,iii] + (sec[any_min[j],:,ii,iii]*molar[any_min[j]]/density[any_min[j]])
            alt_vol_d[:,ii,iii] = alt_vol_d[:,ii,iii] + (sec_d[any_min[j],:,ii,iii]*molar[any_min[j]]/density[any_min[j]])
            alt_vol_a[:,ii,iii] = alt_vol_a[:,ii,iii] + (sec_a[any_min[j],:,ii,iii]*molar[any_min[j]]/density[any_min[j]])
            alt_vol_b[:,ii,iii] = alt_vol_b[:,ii,iii] + (sec_b[any_min[j],:,ii,iii]*molar[any_min[j]]/density[any_min[j]])

            if j == len(any_min) - 1:
                alt_vol[:,ii,iii] = alt_vol[:,ii,iii]/pri[:,ii,iii]
                alt_vol_d[:,ii,iii] = alt_vol_d[:,ii,iii]/pri_d[:,ii,iii]
                alt_vol_a[:,ii,iii] = alt_vol_a[:,ii,iii]/pri_a[:,ii,iii]
                alt_vol_b[:,ii,iii] = alt_vol_b[:,ii,iii]/pri_b[:,ii,iii]

        #todo: fe_vol data

        feo_temp = np.zeros(tp)
        feot_temp = np.zeros(tp)

        feo_temp = 0.166*pri[:,ii,iii] + sec[14,:,ii,iii]
        feot_temp = 0.8998*.026*2.0*pri[:,ii,iii] + 0.8998*sec[7,:,ii,iii] + 2.0*0.8998*sec[17,:,ii,iii] + 2.0*0.8998*sec[13,:,ii,iii] + 2.0*0.8998*sec[15,:,ii,iii] + 2.0*0.8998*sec[12,:,ii,iii]

        alt_fe[:,ii,iii] = feo_temp / (feot_temp + feo_temp)


        feo_temp = 0.166*pri_d[:,ii,iii] + sec_d[14,:,ii,iii]
        feot_temp = 0.8998*.026*2.0*pri_d[:,ii,iii] + 0.8998*sec_d[7,:,ii,iii] + 2.0*0.8998*sec_d[17,:,ii,iii] + 2.0*0.8998*sec_d[13,:,ii,iii] + 2.0*0.8998*sec_d[15,:,ii,iii] + 2.0*0.8998*sec_d[12,:,ii,iii]

        alt_fe_d[:,ii,iii] = feo_temp / (feot_temp + feo_temp)



        feo_temp = 0.166*pri_a[:,ii,iii] + sec_a[14,:,ii,iii]
        feot_temp = 0.8998*.026*2.0*pri_a[:,ii,iii] + 0.8998*sec_a[7,:,ii,iii] + 2.0*0.8998*sec_a[17,:,ii,iii] + 2.0*0.8998*sec_a[13,:,ii,iii] + 2.0*0.8998*sec_a[15,:,ii,iii] + 2.0*0.8998*sec_a[12,:,ii,iii]

        alt_fe_a[:,ii,iii] = feo_temp / (feot_temp + feo_temp)


        feo_temp = 0.166*pri_b[:,ii,iii] + sec_b[14,:,ii,iii]
        feot_temp = 0.8998*.026*2.0*pri_b[:,ii,iii] + 0.8998*sec_b[7,:,ii,iii] + 2.0*0.8998*sec_b[17,:,ii,iii] + 2.0*0.8998*sec_b[13,:,ii,iii] + 2.0*0.8998*sec_b[15,:,ii,iii] + 2.0*0.8998*sec_b[12,:,ii,iii]

        alt_fe_b[:,ii,iii] = feo_temp / (feot_temp + feo_temp)







x_elements[-1,:,:,:] = x_elements[-2,:,:,:]
x_pri_elements[-1,:,:,:] = x_pri_elements[-2,:,:,:]

x_elements[0,:,:,:] = x_elements[1,:,:,:]
x_pri_elements[0,:,:,:] = x_pri_elements[1,:,:,:]


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

            value_dsec[any_min[j],ii,iii] = np.mean(dsec[any_min[j],2:,ii,iii])
            value_dsec_d[any_min[j],ii,iii] = np.mean(dsec_d[any_min[j],2:,ii,iii])
            value_dsec_a[any_min[j],ii,iii] = np.mean(dsec_a[any_min[j],2:,ii,iii])
            value_dsec_b[any_min[j],ii,iii] = np.mean(dsec_b[any_min[j],2:,ii,iii])


            #todo: fill value_alt_vol 2D
            value_alt_vol_mean[ii,iii] = np.mean(alt_vol[1:,ii,iii]-alt_vol[:-1,ii,iii])
            value_alt_vol_mean_d[ii,iii] = np.mean(alt_vol_d[1:,ii,iii]-alt_vol_d[:-1,ii,iii])
            value_alt_vol_mean_a[ii,iii] = np.mean(alt_vol_a[1:,ii,iii]-alt_vol_a[:-1,ii,iii])
            value_alt_vol_mean_b[ii,iii] = np.mean(alt_vol_b[1:,ii,iii]-alt_vol_b[:-1,ii,iii])

            #todo: fill value_alt_fe 2D
            value_alt_fe_mean[ii,iii] = np.mean(alt_fe[1:,ii,iii]-alt_fe[:-1,ii,iii])
            value_alt_fe_mean_d[ii,iii] = np.mean(alt_fe_d[1:,ii,iii]-alt_fe_d[:-1,ii,iii])
            value_alt_fe_mean_a[ii,iii] = np.mean(alt_fe_a[1:,ii,iii]-alt_fe_a[:-1,ii,iii])
            value_alt_fe_mean_b[ii,iii] = np.mean(alt_fe_b[1:,ii,iii]-alt_fe_b[:-1,ii,iii])



bar_bins = 4
bar_shrink = 0.9
xlabelpad = -2
clabelpad = 0






#hack: FIGURE: net_uptake_x
fig=plt.figure(figsize=(15.0,8.0))
print "net_uptake_x"

net_uptake_kwargs = dict(lw=1.1)
net_uptake_kwargs_a = dict(lw=1.1, linestyle='-')
net_uptake_kwargs_b = dict(lw=1.1, linestyle='-')
uptake_color_s = '#bd3706'
uptake_color_d = '#073dc7'
uptake_color_a = '#0793c7'
uptake_color_b = '#0fe7e0'
xt_fs = 8


# Mg
ax=fig.add_subplot(3, 4, 1, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_elements[:,6,ii,0]+x_pri_elements[:,6,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.legend(fontsize=8,loc='best',labelspacing=-0.1,columnspacing=0.0)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo Mg total uptake',fontsize=10)


ax=fig.add_subplot(3, 4, 5, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_pri_elements[:,6,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo Mg leached',fontsize=10)


ax=fig.add_subplot(3, 4, 9, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_elements[:,6,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo Mg uptake',fontsize=10)




# Ca
ax=fig.add_subplot(3, 4, 2, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_elements[:,5,ii,0]+x_pri_elements[:,5,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo Ca total uptake',fontsize=10)


ax=fig.add_subplot(3, 4, 6, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_pri_elements[:,5,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo Ca leached',fontsize=10)


ax=fig.add_subplot(3, 4, 10, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_elements[:,5,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo Ca uptake',fontsize=10)




# K
ax=fig.add_subplot(3, 4, 3, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_elements[:,8,ii,0]+x_pri_elements[:,8,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo K total uptake',fontsize=10)


ax=fig.add_subplot(3, 4, 7, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_pri_elements[:,8,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo K leached',fontsize=10)


ax=fig.add_subplot(3, 4, 11, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_elements[:,8,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo K uptake',fontsize=10)




# Fe
ax=fig.add_subplot(3, 4, 4, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_elements[:,9,ii,0]+x_pri_elements[:,9,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo Fe total uptake',fontsize=10)


ax=fig.add_subplot(3, 4, 8, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_pri_elements[:,9,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo Fe leached',fontsize=10)


ax=fig.add_subplot(3, 4, 12, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(x_elements[:,9,ii,0],color=plot_col[ii], label=param_strings[ii], **net_uptake_kwargs)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('solo Fe uptake',fontsize=10)



# ax=fig.add_subplot(3, 4, 2, frameon=True)
#
# for ii in range(len(param_t_diff)):
#     plt.plot(np.arange(steps),x_elements_d[:,5,ii,iii]+x_pri_elements_d[:,5,ii,iii],color=plot_col[ii], label=plot_t_diff_strings[ii], **net_uptake_kwargs)
# plt.plot(range(steps),x_elements[:,5,0,0]+x_pri_elements[:,5,0,0],color=plot_col[len(param_t_diff)],label=plot_t_diff_strings[len(param_t_diff)], **net_uptake_kwargs)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.title('dual vs solo Ca total uptake',fontsize=10)
#
#
#
# ax=fig.add_subplot(3, 4, 3, frameon=True)
#
# for ii in range(len(param_t_diff)):
#     plt.plot(np.arange(steps),x_elements_d[:,8,ii,iii]+x_pri_elements_d[:,8,ii,iii],color=plot_col[ii], label=plot_t_diff_strings[ii], **net_uptake_kwargs)
# plt.plot(range(steps),x_elements[:,8,0,0]+x_pri_elements[:,8,0,0],color=plot_col[len(param_t_diff)],label=plot_t_diff_strings[len(param_t_diff)], **net_uptake_kwargs)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.title('dual vs solo K total uptake',fontsize=10)
#
#
#
# ax=fig.add_subplot(3, 4, 4, frameon=True)
#
# for ii in range(len(param_t_diff)):
#     plt.plot(np.arange(steps),x_elements_d[:,9,ii,iii]+x_pri_elements_d[:,9,ii,iii],color=plot_col[ii], label=plot_t_diff_strings[ii], **net_uptake_kwargs)
# plt.plot(range(steps),x_elements[:,9,0,0]+x_pri_elements[:,9,0,0],color=plot_col[len(param_t_diff)],label=plot_t_diff_strings[len(param_t_diff)], **net_uptake_kwargs)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.title('dual vs solo Fe total uptake',fontsize=10)



plt.subplots_adjust( wspace=0.3 , hspace=0.3)
plt.savefig(outpath+prefix+"x_s_net.png",bbox_inches='tight')








#hack: 2d pri
# primary slope?
print "2d_pri"
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






x_cont = diff_nums#range(len(diff_strings))
y_cont = param_nums#range(len(param_strings))
x_grid, y_grid = np.meshgrid(x_cont,y_cont)
cont_cmap = cm.rainbow
n_cont = 15
cont_skip = 4


#hack: 2d pri CONTOUR
# primary slope?
print "2d_pri contour"
## HALF LINE
value_dpri_mean_d = value_dpri_mean_d/2.0
fig=plt.figure(figsize=(8.0,8.0))

the_s = np.abs(value_dpri_mean)
the_d = np.abs(value_dpri_mean_d)
the_a = np.abs(value_dpri_mean_a)
the_b = np.abs(value_dpri_mean_b)

min_all = np.min(the_s)
if np.min(the_d) < min_all:
    min_all = np.min(the_d)
if np.min(the_a) < min_all:
    min_all = np.min(the_a)
if np.min(the_b) < min_all:
    min_all = np.min(the_b)

max_all = np.max(the_s)
if np.max(the_d) > max_all:
    max_all = np.max(the_d)
if np.max(the_a) > max_all:
    max_all = np.max(the_a)
if np.max(the_b) > max_all:
    max_all = np.max(the_b)

cont_levels = np.linspace(min_all,max_all,num=n_cont,endpoint=True)




ax=fig.add_subplot(2, 2, 1, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = np.abs(value_dpri_mean)
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")


plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('dpri mean',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")




ax=fig.add_subplot(2, 2, 2, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = np.abs(value_dpri_mean_d)
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")

plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('dpri mean d',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")




ax=fig.add_subplot(2, 2, 3, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = np.abs(value_dpri_mean_a)
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")

plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('dpri mean a',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")




ax=fig.add_subplot(2, 2, 4, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = np.abs(value_dpri_mean_b)
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")

plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('dpri mean b',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")





plt.savefig(outpath+prefix+"y_pri_contour.png",bbox_inches='tight')
plt.savefig(outpath+prefix+"zps_pri_contour.eps",bbox_inches='tight')








title_fs = 10

#hack: 2d sec binary
print "2d_sec_bin"
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













#hack: 2d calcite
print "2d_calcite"
fig=plt.figure(figsize=(12.0,9.0))
this_min = 9


the_s = value_sec[this_min,:,:]
the_d = value_sec_d[this_min,:,:]
the_a = value_sec_a[this_min,:,:]
the_b = value_sec_b[this_min,:,:]

min_all = np.min(the_s)
if np.min(the_d) < min_all:
    min_all = np.min(the_d)
if np.min(the_a) < min_all:
    min_all = np.min(the_a)
if np.min(the_b) < min_all:
    min_all = np.min(the_b)

max_all = np.max(the_s)
if np.max(the_d) > max_all:
    max_all = np.max(the_d)
if np.max(the_a) > max_all:
    max_all = np.max(the_a)
if np.max(the_b) > max_all:
    max_all = np.max(the_b)


ax=fig.add_subplot(3, 4, 1, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

plt.xlabel('param_t_diff')
plt.ylabel('param_q')

this_plot = value_sec[this_min,:,:]
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)



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






ax=fig.add_subplot(3, 4, 3, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_sec_a[this_min,:,:]
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)






ax=fig.add_subplot(3, 4, 4, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_sec_b[this_min,:,:]
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)






plt.savefig(outpath+prefix+"x_2d_calcite.png",bbox_inches='tight')

















#hack: 2d sec
print "2d_sec"

#### DIVIDE value_sec_d by 2.0!!!! IMPORTANT
value_sec_d = value_sec_d/2.0

fig=plt.figure(figsize=(20.0,len(any_min)))
sp1 = (len(any_min)+1)/2
sp2 = 8
plt.subplots_adjust(hspace=0.5)
y_off = 0.02
print "len(any_min): " , len(any_min)
print "len(any_min)/2: " , len(any_min)/2

for j in range(len(any_min)):


    ## 2d sec calcite
    this_min = any_min[j]


    the_s = value_sec[this_min,:,:]
    the_d = value_sec_d[this_min,:,:]
    the_a = value_sec_a[this_min,:,:]
    the_b = value_sec_b[this_min,:,:]

    min_all = np.min(the_s)
    if np.min(the_d) < min_all:
        min_all = np.min(the_d)
    if np.min(the_a) < min_all:
        min_all = np.min(the_a)
    if np.min(the_b) < min_all:
        min_all = np.min(the_b)

    max_all = np.max(the_s)
    if np.max(the_d) > max_all:
        max_all = np.max(the_d)
    if np.max(the_a) > max_all:
        max_all = np.max(the_a)
    if np.max(the_b) > max_all:
        max_all = np.max(the_b)

    sp_factor = (j*4)
    if j == 0:
        sp_factor = 0


    ax=fig.add_subplot(sp1, sp2, sp_factor+1, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    # plt.xlabel('param_t_diff')
    # plt.ylabel('param_q')

    this_plot = value_sec[this_min,:,:]
    plt.title(secondary[any_min[j]], fontsize=10)
    if np.max(this_plot) > 0.0:
        this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
        pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)







    ax=fig.add_subplot(sp1, sp2, sp_factor+2, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    this_plot = value_sec_d[this_min,:,:]
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

    bbox = ax.get_position()
    cax = fig.add_axes([bbox.xmin+0.0, bbox.ymin-y_off, bbox.width*1.5, bbox.height*0.06])
    cbar = plt.colorbar(pCol, cax = cax,orientation='horizontal')
    cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
    cbar.ax.tick_params(labelsize=7)
    #plt.title('CaCO3 at end',fontsize=9)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")






    ax=fig.add_subplot(sp1, sp2, sp_factor+3, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    this_plot = value_sec_a[this_min,:,:]
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    plt.pcolor(this_plot, vmin=min_all, vmax=max_all)






    ax=fig.add_subplot(sp1, sp2, sp_factor+4, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    this_plot = value_sec_b[this_min,:,:]
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    plt.pcolor(this_plot, vmin=min_all, vmax=max_all)









plt.savefig(outpath+prefix+"x_2d_sec.png",bbox_inches='tight')










#hack: 2d sec CONTOUR
print "2d sec contour"
fig=plt.figure(figsize=(20.0,len(any_min)))
sp1 = (len(any_min)+1)/2
sp2 = 8
plt.subplots_adjust(hspace=0.5)
y_off = 0.03

cont_x_skip = 2

for j in range(len(any_min)):


    ## 2d sec calcite
    this_min = any_min[j]


    the_s = value_sec[this_min,:,:]
    the_d = value_sec_d[this_min,:,:]
    the_a = value_sec_a[this_min,:,:]/2.0
    the_b = value_sec_b[this_min,:,:]/2.0

    min_all = np.min(the_s)
    if np.min(the_d) < min_all:
        min_all = np.min(the_d)
    if np.min(the_a) < min_all:
        min_all = np.min(the_a)
    if np.min(the_b) < min_all:
        min_all = np.min(the_b)

    max_all = np.max(the_s)
    if np.max(the_d) > max_all:
        max_all = np.max(the_d)
    if np.max(the_a) > max_all:
        max_all = np.max(the_a)
    if np.max(the_b) > max_all:
        max_all = np.max(the_b)

    cont_levels = np.linspace(min_all,max_all,num=n_cont,endpoint=True)

    sp_factor = (j*4)
    if j == 0:
        sp_factor = 0



    ax=fig.add_subplot(sp1, sp2, sp_factor+1, frameon=True)

    plt.xticks(diff_nums[::cont_x_skip],diff_strings[::cont_x_skip], fontsize=8)
    plt.yticks(param_nums,param_strings, fontsize=8)

    # plt.xlabel('log10(mixing time [years])')
    # plt.ylabel('discharge q [m/yr]')

    this_plot = value_sec[this_min,:,:]
    plt.title(secondary[any_min[j]], fontsize=10)
    if np.max(this_plot) > 0.0:
        this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
        pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
        for c in pCont.collections:
            c.set_edgecolor("face")







    ax=fig.add_subplot(sp1, sp2, sp_factor+2, frameon=True)

    plt.xticks(diff_nums[::cont_x_skip],diff_strings[::cont_x_skip], fontsize=8)
    plt.yticks(param_nums,param_strings, fontsize=8)

    this_plot = value_sec_d[this_min,:,:]
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
    for c in pCont.collections:
        c.set_edgecolor("face")

    bbox = ax.get_position()
    cax = fig.add_axes([bbox.xmin+0.0, bbox.ymin-y_off, bbox.width*1.5, bbox.height*0.06])
    cbar = plt.colorbar(pCont, cax = cax,orientation='horizontal')
    cbar.set_ticks(cont_levels[::cont_skip])
    cbar.ax.tick_params(labelsize=7)
    #plt.title('CaCO3 at end',fontsize=9)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")






    ax=fig.add_subplot(sp1, sp2, sp_factor+3, frameon=True)

    plt.xticks(diff_nums[::cont_x_skip],diff_strings[::cont_x_skip], fontsize=8)
    plt.yticks(param_nums,param_strings, fontsize=8)

    this_plot = value_sec_a[this_min,:,:]/2.0
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
    for c in pCont.collections:
        c.set_edgecolor("face")






    ax=fig.add_subplot(sp1, sp2, sp_factor+4, frameon=True)

    plt.xticks(diff_nums[::cont_x_skip],diff_strings[::cont_x_skip], fontsize=8)
    plt.yticks(param_nums,param_strings, fontsize=8)

    this_plot = value_sec_b[this_min,:,:]/2.0
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
    for c in pCont.collections:
        c.set_edgecolor("face")




plt.savefig(outpath+prefix+"y_sec_contour.png",bbox_inches='tight')
plt.savefig(outpath+prefix+"zps_sec_contour.eps",bbox_inches='tight')






















#hack: 2d sec rate
print "2d_sec_rate"

#### DIVIDE value_dsec_d by 2.0!!!! IMPORTANT
value_dsec_d = value_dsec_d/2.0

fig=plt.figure(figsize=(20.0,len(any_min)))
sp1 = (len(any_min)+1)/2
sp2 = 8
plt.subplots_adjust(hspace=0.5)
y_off = 0.02
print "len(any_min): " , len(any_min)
print "len(any_min)/2: " , len(any_min)/2

for j in range(len(any_min)):


    ## 2d sec calcite
    this_min = any_min[j]


    the_s = value_dsec[this_min,:,:]
    the_d = value_dsec_d[this_min,:,:]
    the_a = value_dsec_a[this_min,:,:]
    the_b = value_dsec_b[this_min,:,:]

    min_all = np.min(the_s)
    if np.min(the_d) < min_all:
        min_all = np.min(the_d)
    if np.min(the_a) < min_all:
        min_all = np.min(the_a)
    if np.min(the_b) < min_all:
        min_all = np.min(the_b)

    max_all = np.max(the_s)
    if np.max(the_d) > max_all:
        max_all = np.max(the_d)
    if np.max(the_a) > max_all:
        max_all = np.max(the_a)
    if np.max(the_b) > max_all:
        max_all = np.max(the_b)

    sp_factor = (j*4)
    if j == 0:
        sp_factor = 0


    ax=fig.add_subplot(sp1, sp2, sp_factor+1, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    # plt.xlabel('param_t_diff')
    # plt.ylabel('param_q')

    this_plot = value_dsec[this_min,:,:]
    plt.title(secondary[any_min[j]]+" rate!", fontsize=10)
    if np.max(this_plot) > 0.0:
        this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
        pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)






    ax=fig.add_subplot(sp1, sp2, sp_factor+2, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    this_plot = value_dsec_d[this_min,:,:]
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

    bbox = ax.get_position()
    cax = fig.add_axes([bbox.xmin+0.0, bbox.ymin-y_off, bbox.width*1.5, bbox.height*0.06])
    cbar = plt.colorbar(pCol, cax = cax,orientation='horizontal')
    cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
    cbar.ax.tick_params(labelsize=7)
    #plt.title('CaCO3 at end',fontsize=9)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")






    ax=fig.add_subplot(sp1, sp2, sp_factor+3, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    this_plot = value_dsec_a[this_min,:,:]
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    plt.pcolor(this_plot, vmin=min_all, vmax=max_all)






    ax=fig.add_subplot(sp1, sp2, sp_factor+4, frameon=True)

    plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
    plt.yticks(the_yticks,param_strings, fontsize=8)

    this_plot = value_dsec_b[this_min,:,:]
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




plt.savefig(outpath+prefix+"x_2d_sec_rate.png",bbox_inches='tight')











#hack: 2d dsec CONTOUR
print "2d dsec contour"
fig=plt.figure(figsize=(20.0,len(any_min)))
sp1 = (len(any_min)+1)/2
sp2 = 8
plt.subplots_adjust(hspace=0.5)
y_off = 0.03

cont_x_skip = 2

for j in range(len(any_min)):


    ## 2d sec calcite
    this_min = any_min[j]


    the_s = value_dsec[this_min,:,:]
    the_d = value_dsec_d[this_min,:,:]
    the_a = value_dsec_a[this_min,:,:]/2.0
    the_b = value_dsec_b[this_min,:,:]/2.0

    min_all = np.min(the_s)
    if np.min(the_d) < min_all:
        min_all = np.min(the_d)
    if np.min(the_a) < min_all:
        min_all = np.min(the_a)
    if np.min(the_b) < min_all:
        min_all = np.min(the_b)

    max_all = np.max(the_s)
    if np.max(the_d) > max_all:
        max_all = np.max(the_d)
    if np.max(the_a) > max_all:
        max_all = np.max(the_a)
    if np.max(the_b) > max_all:
        max_all = np.max(the_b)

    cont_levels = np.linspace(min_all,max_all,num=n_cont,endpoint=True)

    sp_factor = (j*4)
    if j == 0:
        sp_factor = 0



    ax=fig.add_subplot(sp1, sp2, sp_factor+1, frameon=True)

    plt.xticks(diff_nums[::cont_x_skip],diff_strings[::cont_x_skip], fontsize=8)
    plt.yticks(param_nums,param_strings, fontsize=8)

    # plt.xlabel('log10(mixing time [years])')
    # plt.ylabel('discharge q [m/yr]')

    this_plot = value_dsec[this_min,:,:]
    plt.title(secondary[any_min[j]], fontsize=10)
    if np.max(this_plot) > 0.0:
        this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
        pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
        for c in pCont.collections:
            c.set_edgecolor("face")





    ax=fig.add_subplot(sp1, sp2, sp_factor+2, frameon=True)

    plt.xticks(diff_nums[::cont_x_skip],diff_strings[::cont_x_skip], fontsize=8)
    plt.yticks(param_nums,param_strings, fontsize=8)

    this_plot = value_dsec_d[this_min,:,:]
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
    for c in pCont.collections:
        c.set_edgecolor("face")

    bbox = ax.get_position()
    cax = fig.add_axes([bbox.xmin+0.0, bbox.ymin-y_off, bbox.width*1.5, bbox.height*0.06])
    cbar = plt.colorbar(pCont, cax = cax,orientation='horizontal')
    cbar.set_ticks(cont_levels[::cont_skip])
    cbar.ax.tick_params(labelsize=7)
    #plt.title('CaCO3 at end',fontsize=9)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")




    ax=fig.add_subplot(sp1, sp2, sp_factor+3, frameon=True)

    plt.xticks(diff_nums[::cont_x_skip],diff_strings[::cont_x_skip], fontsize=8)
    plt.yticks(param_nums,param_strings, fontsize=8)

    this_plot = value_dsec_a[this_min,:,:]/2.0
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
    for c in pCont.collections:
        c.set_edgecolor("face")




    ax=fig.add_subplot(sp1, sp2, sp_factor+4, frameon=True)

    plt.xticks(diff_nums[::cont_x_skip],diff_strings[::cont_x_skip], fontsize=8)
    plt.yticks(param_nums,param_strings, fontsize=8)

    this_plot = value_dsec_b[this_min,:,:]/2.0
    this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
    pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
    for c in pCont.collections:
        c.set_edgecolor("face")





plt.savefig(outpath+prefix+"y_dsec_contour.png",bbox_inches='tight')
plt.savefig(outpath+prefix+"zps_dsec_contour.eps",bbox_inches='tight')













#hack: 2d dsol
print "2d_dsol"
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













#hack: 2d alt_vol
print "2d_alt_vol"
fig=plt.figure(figsize=(9.0,5))
plt.subplots_adjust(hspace=0.5)

sp1 = 2
sp2 = 4


the_s = value_alt_vol_mean
the_d = value_alt_vol_mean_d
the_a = value_alt_vol_mean_a
the_b = value_alt_vol_mean_b

min_all = np.min(the_s)
if np.min(the_d) < min_all:
    min_all = np.min(the_d)
if np.min(the_a) < min_all:
    min_all = np.min(the_a)
if np.min(the_b) < min_all:
    min_all = np.min(the_b)

max_all = np.max(the_s)
if np.max(the_d) > max_all:
    max_all = np.max(the_d)
if np.max(the_a) > max_all:
    max_all = np.max(the_a)
if np.max(the_b) > max_all:
    max_all = np.max(the_b)

"alt vol min + max"
print min_all
print max_all
print " "


ax=fig.add_subplot(sp1, sp2, 1, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

plt.xlabel('param_t_diff')
plt.ylabel('param_q')

this_plot = value_alt_vol_mean
pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

bbox = ax.get_position()
cax = fig.add_axes([bbox.xmin+0.25, bbox.ymin-0.1, bbox.width*1.5, bbox.height*0.06])
cbar = plt.colorbar(pCol, cax = cax,orientation='horizontal')
cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
cbar.ax.tick_params(labelsize=7)
plt.title('alt vol slope mean',fontsize=9)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")


ax=fig.add_subplot(sp1, sp2, 2, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_alt_vol_mean_d
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 3, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_alt_vol_mean_a
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 4, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_alt_vol_mean_b
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)










the_s = value_alt_fe_mean
the_d = value_alt_fe_mean_d
the_a = value_alt_fe_mean_a
the_b = value_alt_fe_mean_b

min_all = np.min(the_s)
if np.min(the_d) < min_all:
    min_all = np.min(the_d)
if np.min(the_a) < min_all:
    min_all = np.min(the_a)
if np.min(the_b) < min_all:
    min_all = np.min(the_b)

max_all = np.max(the_s)
if np.max(the_d) > max_all:
    max_all = np.max(the_d)
if np.max(the_a) > max_all:
    max_all = np.max(the_a)
if np.max(the_b) > max_all:
    max_all = np.max(the_b)

print "alt_fe min + max"
print min_all
print max_all
print " "


ax=fig.add_subplot(sp1, sp2, 5, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

plt.xlabel('param_t_diff')
plt.ylabel('param_q')

this_plot = value_alt_fe_mean
print "solo value_alt_fe_mean"
print this_plot
pCol = plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

bbox = ax.get_position()
cax = fig.add_axes([bbox.xmin+0.25, bbox.ymin-0.1, bbox.width*1.5, bbox.height*0.06])
cbar = plt.colorbar(pCol, cax = cax,orientation='horizontal')
cbar.set_ticks(np.linspace(min_all,max_all,num=bar_bins,endpoint=True))
cbar.ax.tick_params(labelsize=7)
plt.title('alt fe slope mean',fontsize=9)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")


ax=fig.add_subplot(sp1, sp2, 6, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_alt_fe_mean_d
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 7, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_alt_fe_mean_a
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




ax=fig.add_subplot(sp1, sp2, 8, frameon=True)

plt.xticks(the_xticks[::2],diff_strings[::2], fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

this_plot = value_alt_fe_mean_b
this_plot = np.ma.masked_where(this_plot == 0.0, this_plot)
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)




#
plt.savefig(outpath+prefix+"x_2d_alt_vol.png",bbox_inches='tight')














cont_cmap = cm.rainbow
n_cont = 15
cont_skip = 4

sp1 = 2
sp2 = 4


#hack: 2d alt_vol CONTOUR
# primary slope?
print "2d_alt_vol contour"
## HALF LINE
#####value_dpri_mean_d = value_dpri_mean_d/2.0
fig=plt.figure(figsize=(12.0,8.0))



the_s = value_alt_vol_mean
the_d = value_alt_vol_mean_d
the_a = value_alt_vol_mean_a
the_b = value_alt_vol_mean_b

min_all = np.min(the_s)
if np.min(the_d) < min_all:
    min_all = np.min(the_d)
if np.min(the_a) < min_all:
    min_all = np.min(the_a)
if np.min(the_b) < min_all:
    min_all = np.min(the_b)

max_all = np.max(the_s)
if np.max(the_d) > max_all:
    max_all = np.max(the_d)
if np.max(the_a) > max_all:
    max_all = np.max(the_a)
if np.max(the_b) > max_all:
    max_all = np.max(the_b)

cont_levels = np.linspace(min_all,max_all,num=n_cont,endpoint=True)




ax=fig.add_subplot(sp1, sp2, 1, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = value_alt_vol_mean
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")


plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('alt_vol slope mean',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")




ax=fig.add_subplot(sp1, sp2, 2, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = value_alt_vol_mean_d
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")

plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('alt_vol slope mean d',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")




ax=fig.add_subplot(sp1, sp2, 3, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = value_alt_vol_mean_a
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")

plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('alt_vol slope mean a',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")




ax=fig.add_subplot(sp1, sp2, 4, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = value_alt_vol_mean_b
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")

plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('alt_vol slope mean b',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")














the_s = value_alt_fe_mean
the_d = value_alt_fe_mean_d
the_a = value_alt_fe_mean_a
the_b = value_alt_fe_mean_b

min_all = np.min(the_s)
if np.min(the_d) < min_all:
    min_all = np.min(the_d)
if np.min(the_a) < min_all:
    min_all = np.min(the_a)
if np.min(the_b) < min_all:
    min_all = np.min(the_b)

max_all = np.max(the_s)
if np.max(the_d) > max_all:
    max_all = np.max(the_d)
if np.max(the_a) > max_all:
    max_all = np.max(the_a)
if np.max(the_b) > max_all:
    max_all = np.max(the_b)

cont_levels = np.linspace(min_all,max_all,num=n_cont,endpoint=True)




ax=fig.add_subplot(sp1, sp2, 5, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = value_alt_fe_mean
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")


plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('FeO/FeOt slope mean',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")




ax=fig.add_subplot(sp1, sp2, 6, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = value_alt_fe_mean_d
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")

plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('FeO/FeOt slope mean d',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")




ax=fig.add_subplot(sp1, sp2, 7, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = value_alt_fe_mean_a
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")

plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('FeO/FeOt slope mean a',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")




ax=fig.add_subplot(sp1, sp2, 8, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = value_alt_fe_mean_b
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")

plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('FeO/FeOt slope mean b',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")


plt.savefig(outpath+prefix+"y_alt_vol_contour.png",bbox_inches='tight')
plt.savefig(outpath+prefix+"zps_alt_vol_contour.eps",bbox_inches='tight')





















#hack: alk 2d plot
print "2d_alk"
fig=plt.figure(figsize=(8.0,8.0))

value_alk_mean_d = value_alk_mean_d/2.0

# min_s_d = np.min(value_alk_mean)
# if np.min(value_alk_mean_d) < min_s_d:
#     min_s_d = np.min(value_alk_mean_d)
#
# max_s_d = np.max(value_alk_mean)
# if np.max(value_alk_mean_d) > max_s_d:
#     max_s_d = np.max(value_alk_mean_d)


the_s = value_alk_mean
the_d = value_alk_mean_d
the_a = value_alk_mean_a
the_b = value_alk_mean_b

min_all = np.min(the_s)
if np.min(the_d) < min_all:
    min_all = np.min(the_d)
# if np.min(the_a) < min_all:
#     min_all = np.min(the_a)
# if np.min(the_b) < min_all:
#     min_all = np.min(the_b)

max_all = np.max(the_s)
if np.max(the_d) > max_all:
    max_all = np.max(the_d)
# if np.max(the_a) > max_all:
#     max_all = np.max(the_a)
# if np.max(the_b) > max_all:
#     max_all = np.max(the_b)




ax=fig.add_subplot(2, 2, 1, frameon=True)

plt.xticks(the_xticks,diff_strings, fontsize=8)
plt.yticks(the_yticks,param_strings, fontsize=8)

plt.xlabel('param_t_diff')
plt.ylabel('param_q')

this_plot = value_alk_mean
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

this_plot_pos = np.ma.masked_where(this_plot < 0.0, this_plot)
plt.pcolor(this_plot_pos, vmin=min_all, vmax=max_all, edgecolors='black', linewidths='1')
print "this_plot_pos"
print this_plot_pos
print " "




cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks([min_all,max_all,0.0])
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
plt.pcolor(this_plot, vmin=min_all, vmax=max_all)

this_plot_pos = np.ma.masked_where(this_plot < 0.0, this_plot)
plt.pcolor(this_plot_pos, vmin=min_all, vmax=max_all, edgecolors='black', linewidths='1')

cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks([min_all,max_all,0.0])
#cbar.set_ticks([np.min(this_plot),np.max(this_plot),0.0])
#cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
cbar.ax.set_xlabel('alk_d mean',fontsize=10,labelpad=clabelpad)


print "value_alk_mean_d"
print value_alk_mean_d
print " "


# ax=fig.add_subplot(2, 2, 3, frameon=True)
#
# plt.xticks(the_xticks,diff_strings, fontsize=8)
# plt.yticks(the_yticks,param_strings, fontsize=8)
#
# this_plot = value_alk_mean_a
# plt.pcolor(this_plot, vmin=min_s_d, vmax=max_s_d)
#
# this_plot_pos = np.ma.masked_where(this_plot < 0.0, this_plot)
# plt.pcolor(this_plot_pos, vmin=min_s_d, vmax=max_s_d, edgecolors='black', linewidths='1')
#
# cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
# cbar.ax.tick_params(labelsize=8)
# cbar.set_ticks([np.min(this_plot),np.max(this_plot),0.0])
# #cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
# cbar.ax.set_xlabel('alk_a mean',fontsize=10,labelpad=clabelpad)
#
#
#
#
# ax=fig.add_subplot(2, 2, 4, frameon=True)
#
# plt.xticks(the_xticks,diff_strings, fontsize=8)
# plt.yticks(the_yticks,param_strings, fontsize=8)
#
# this_plot = value_alk_mean_b
# plt.pcolor(this_plot, vmin=min_s_d, vmax=max_s_d)
#
# this_plot_pos = np.ma.masked_where(this_plot < 0.0, this_plot)
# plt.pcolor(this_plot_pos, vmin=min_s_d, vmax=max_s_d, edgecolors='black', linewidths='1')
#
# cbar = plt.colorbar(orientation='horizontal',shrink=bar_shrink)
# cbar.ax.tick_params(labelsize=8)
# cbar.set_ticks([np.min(this_plot),np.max(this_plot),0.0])
# #cbar.set_ticks(np.linspace(np.min(this_plot[this_plot>0.0]),np.max(this_plot),num=bar_bins,endpoint=True))
# cbar.ax.set_xlabel('alk_b mean',fontsize=10,labelpad=clabelpad)


plt.savefig(outpath+prefix+"x_2d_alk.png",bbox_inches='tight')











cont_cmap = cm.rainbow
n_cont = 15
cont_skip = 4


#hack: alk 2d contour
# primary slope?
print "2d_alk contour"

fig=plt.figure(figsize=(8.0,8.0))

the_s = value_alk_mean
the_d = value_alk_mean_d
the_a = value_alk_mean_a
the_b = value_alk_mean_b

min_all = np.min(the_s)
if np.min(the_d) < min_all:
    min_all = np.min(the_d)
# if np.min(the_a) < min_all:
#     min_all = np.min(the_a)
# if np.min(the_b) < min_all:
#     min_all = np.min(the_b)

max_all = np.max(the_s)
if np.max(the_d) > max_all:
    max_all = np.max(the_d)
# if np.max(the_a) > max_all:
#     max_all = np.max(the_a)
# if np.max(the_b) > max_all:
#     max_all = np.max(the_b)

cont_levels = np.linspace(min_all,max_all,num=n_cont,endpoint=True)




ax=fig.add_subplot(2, 2, 1, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = value_alk_mean
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")


plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('dpri mean',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")

pCont = plt.contour(x_grid,y_grid,this_plot, levels=[0.0,1.0], colors='w', linewidth=4.0)




ax=fig.add_subplot(2, 2, 2, frameon=True)

plt.xlabel('log10(mixing time [years])')
plt.ylabel('discharge q [m/yr]')

this_plot = value_alk_mean_d
pCont = plt.contourf(x_grid,y_grid,this_plot, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
for c in pCont.collections:
    c.set_edgecolor("face")

plt.xticks(diff_nums,diff_strings, fontsize=8)
plt.yticks(param_nums,param_strings, fontsize=8)

cbar = plt.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(cont_levels[::cont_skip])
cbar.ax.set_xlabel('dpri mean d',fontsize=10,labelpad=clabelpad)
cbar.solids.set_rasterized(True)
cbar.solids.set_edgecolor("face")


pCont = plt.contour(x_grid,y_grid,this_plot, levels=[0.0,1.0], colors='w', linewidth=4.0)






plt.savefig(outpath+prefix+"y_alk_contour.png",bbox_inches='tight')
plt.savefig(outpath+prefix+"zps_alk_contour.eps",bbox_inches='tight')











#hack: plot pri groups
print "pri_groups"
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
print "plot sec"
fig=plt.figure(figsize=(18.0,6))


for j in range(len(any_min)):
    ax=fig.add_subplot(3, 7, j+1, frameon=True)
    this_min = any_min[j]
    for ii in range(len(param_strings)):
    # for ii in [4]:
        plt.plot(sec[this_min,:,ii,0], label=param_strings[ii], c=plot_col[ii])
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
print "plot alk"
fig=plt.figure(figsize=(12.0,6.0))

# FIRST ROW: alk x 3 for ch_s
ax=fig.add_subplot(2, 4, 1, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(alk_out[1:,ii,0] - alk_in[1:,ii,0], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.legend(fontsize=8,bbox_to_anchor=(1.2, 1.2),ncol=2,labelspacing=0.1,columnspacing=0.1)
plt.title('alk_out - alk_in solo')


ax=fig.add_subplot(2, 4, 2, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(alk_in[1:,ii,0], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('alk_in solo')


ax=fig.add_subplot(2, 4, 3, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(alk_out[1:,ii,0], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('alk_out solo')


ax=fig.add_subplot(2, 4, 4, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(alk_out[1:,ii,0], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
    plt.plot(alk_in[1:,ii,0], label=param_strings[ii], c=plot_col[ii], lw=the_lw, linestyle='--')
plt.title('alk_out solid, alk_in dashed')


ax=fig.add_subplot(2, 4, 5, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(sol[2,1:,ii,0], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('[alk] solo')


ax=fig.add_subplot(2, 4, 6, frameon=True)
for ii in range(len(param_strings)):
    plt.plot(sol[1,1:,ii,0], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
plt.title('pH solo')


# ax=fig.add_subplot(2, 4, 7, frameon=True)
# for ii in range(len(param_strings)):
#     plt.plot(alk_vol_out[1:,ii,0] - alk_vol_in[1:,ii,0], label=param_strings[ii], c=plot_col[ii], lw=the_lw)
# plt.title('alk vol solo?')



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
