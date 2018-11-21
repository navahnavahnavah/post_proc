# bb_30_60.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os.path
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 11
# plt.rcParams['hatch.linewidth'] = 2.0
from matplotlib.colors import LinearSegmentedColormap

plot_col = ['#000000', '#940000', '#d26618', '#dfa524', '#9ac116', '#139a31', '#35b5aa', '#0740d2', '#7f05d4', '#b100de', '#fba8ff']
plot_col_hatch = ['#000000', '#690a0a', '#984408', '#af7e11', '#7a9a0e', '#137c2a', '#35b5aa', '#0740d2', '#7f05d4', '#b100de', '#fba8ff']

col = ['#6e0202', '#fc385b', '#ff7411', '#19a702', '#00520d', '#00ffc2', '#609ff2', '#20267c','#8f00ff', '#ec52ff', '#6e6e6e', '#000000', '#df9a00', '#7d4e22', '#ffff00', '#df9a00', '#812700', '#6b3f67', '#0f9995', '#4d4d4d', '#d9d9d9', '#e9acff']



def square_pcolor(sp1, sp2, sp, pcolor_block, cb_title="", xlab=0, ylab=0, the_cbar=0, min_all_in=0.0, max_all_in=1.0):
    ax2=fig.add_subplot(sp1, sp2, sp, frameon=True)

    if xlab == 1:
        plt.xlabel('log10(mixing time [years])', fontsize=7)
    if ylab == 11:
        plt.ylabel('discharge q [m/yr]', fontsize=7)

    pCol = ax2.pcolor(pcolor_block, cmap=cont_cmap, antialiased=True, vmin=min_all_in, vmax=max_all_in)

    x_integers = range(len(diff_nums[:cont_x_diff_max]))
    for x_i in range(len(x_integers)):
        x_integers[x_i] = x_integers[x_i] + 0.5

    y_integers = range(len(param_nums[:cont_y_param_max]))
    for y_i in range(len(y_integers)):
        y_integers[y_i] = y_integers[y_i] + 0.5

    plt.xlim([np.min(x_integers)-0.5,np.max(x_integers)+0.5])

    plt.xticks(x_integers[::xskip],diff_strings[::xskip], fontsize=8)
    plt.yticks(y_integers[::yskip],param_strings[::yskip], fontsize=8)

    plt.title(cb_title, fontsize=8)

    if the_cbar == 1:
        bbox = ax2.get_position()
        # cax = fig.add_axes([bbox.xmin+0.0, bbox.ymin-0.06, bbox.width*1.0, bbox.height*0.05])
        cax = fig.add_axes([bbox.xmin-0.04, bbox.ymin-0.0, bbox.width*0.07, bbox.height*1.05])
        cbar = plt.colorbar(pCol, cax = cax,orientation='vertical')
        cax.yaxis.set_ticks_position('left')
        cbar.set_ticks(np.linspace(min_all_in,max_all_in,num=bar_bins,endpoint=True))
        cbar.ax.tick_params(labelsize=7)
        #cbar.ax.set_xlabel(cb_title,fontsize=9,labelpad=clabelpad)
        cbar.solids.set_edgecolor("face")

    return square_pcolor




def square_pcolor_min(sp1, sp2, sp, pcolor_block, cb_title="", xlab=0, ylab=0, the_cbar=0, min_all_in=0.0, max_all_in=1.0):
    ax2=fig.add_subplot(sp1, sp2, sp, frameon=True)

    if xlab == 1:
        plt.xlabel('log10(mixing time [years])', fontsize=7)
    if ylab == 1:
        plt.ylabel('discharge q [m/yr]', fontsize=7)

    pCol = ax2.pcolor(pcolor_block, cmap=cont_cmap, antialiased=True, vmin=min_all_in, vmax=max_all_in)

    x_integers = range(len(diff_nums[:cont_x_diff_max]))
    for x_i in range(len(x_integers)):
        x_integers[x_i] = x_integers[x_i] + 0.5

    y_integers = range(len(param_nums[:cont_y_param_max]))
    for y_i in range(len(y_integers)):
        y_integers[y_i] = y_integers[y_i] + 0.5

    plt.xlim([np.min(x_integers)-0.5,np.max(x_integers)+0.5])

    plt.xticks(x_integers[::xskip],diff_strings[::xskip], fontsize=7)
    plt.yticks(y_integers[::yskip],param_strings[::yskip], fontsize=7)

    plt.title(cb_title, fontsize=9)

    if the_cbar == 1:
        bbox = ax2.get_position()
        cax = fig.add_axes([bbox.xmin+0.15, bbox.ymin-0.02, bbox.width*2.0, bbox.height*0.06])
        cbar = plt.colorbar(pCol, cax = cax,orientation='horizontal')
        cbar.set_ticks(np.linspace(min_all_in,max_all_in,num=bar_bins,endpoint=True))
        cbar.ax.tick_params(labelsize=7)
        #cbar.ax.set_xlabel(cb_title,fontsize=7,labelpad=clabelpad)
        cbar.solids.set_edgecolor("face")

    return square_pcolor_min







def any_2d_interp(x_in, y_in, z_in, x_diff_path, y_param_path, kind_in='linear'):

    the_f = interpolate.interp2d(x_in, y_in, z_in, kind=kind_in)
    any_2d_interp = the_f(x_diff_path, y_param_path)

    return any_2d_interp


#todo: path + params
temp_string = "65"
temp_string_list = ['30', '35', '40', '45', '50', '55', '60', '65']
temp_int_list = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0]
in_path = "../output/revival/winter_broken/"

if not os.path.exists(in_path+"fig_lateral_final"):
    os.makedirs(in_path+"fig_lateral_final")
brk_path = in_path+"fig_lateral_final/"



param_strings = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']
param_nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

diff_strings = ['2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50', '3.75', '4.00', '4.25', '4.50', '4.75', '5.00', '5.25', '5.50', '5.75', '6.00']
diff_nums = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0]

#poop: make 2d alt_ind grids
n_grids = 8


value_alt_vol_mean = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_vol_mean_d = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_vol_mean_a = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_vol_mean_b = np.zeros([len(param_strings),len(diff_strings),n_grids])

value_alt_fe_mean = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_fe_mean_d = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_fe_mean_a = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_fe_mean_b = np.zeros([len(param_strings),len(diff_strings),n_grids])

value_dpri_mean = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_dpri_mean_d = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_dpri_mean_a = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_dpri_mean_b = np.zeros([len(param_strings),len(diff_strings),n_grids])


#hack: make save arrays JUNE
save_feot_mean = np.zeros([len(param_strings),len(diff_strings),n_grids])
save_feot_mean_d = np.zeros([len(param_strings),len(diff_strings),n_grids])
save_feot_mean_a = np.zeros([len(param_strings),len(diff_strings),n_grids])
save_feot_mean_b = np.zeros([len(param_strings),len(diff_strings),n_grids])

save_mgo_mean = np.zeros([len(param_strings),len(diff_strings),n_grids])
save_mgo_mean_d = np.zeros([len(param_strings),len(diff_strings),n_grids])
save_mgo_mean_a = np.zeros([len(param_strings),len(diff_strings),n_grids])
save_mgo_mean_b = np.zeros([len(param_strings),len(diff_strings),n_grids])

save_k2o_mean = np.zeros([len(param_strings),len(diff_strings),n_grids])
save_k2o_mean_d = np.zeros([len(param_strings),len(diff_strings),n_grids])
save_k2o_mean_a = np.zeros([len(param_strings),len(diff_strings),n_grids])
save_k2o_mean_b = np.zeros([len(param_strings),len(diff_strings),n_grids])

#poop: make value_sec_x grids
minNum = 41
value_dsec = np.zeros([len(param_strings),len(diff_strings),minNum+1,n_grids])
value_dsec_d = np.zeros([len(param_strings),len(diff_strings),minNum+1,n_grids])
value_dsec_a = np.zeros([len(param_strings),len(diff_strings),minNum+1,n_grids])
value_dsec_b = np.zeros([len(param_strings),len(diff_strings),minNum+1,n_grids])


#poop: make alt_ind age curves
curve_nsteps = 1000
n_curves = 10


#poop: max lims go here
cont_x_diff_max = len(diff_strings) - 0
cont_y_param_max = len(param_strings) - 0

alt_vol_curve = np.zeros([curve_nsteps,n_curves])
alt_vol_curve_d = np.zeros([curve_nsteps,n_curves])
alt_vol_curve_a = np.zeros([curve_nsteps,n_curves])
alt_vol_curve_b = np.zeros([curve_nsteps,n_curves])

alt_vol_curve_diff_d = np.zeros([curve_nsteps,n_curves,len(diff_nums)])
alt_vol_curve_diff_a = np.zeros([curve_nsteps,n_curves,len(diff_nums)])
alt_vol_curve_diff_b = np.zeros([curve_nsteps,n_curves,len(diff_nums)])

alt_fe_curve = np.zeros([curve_nsteps,n_curves])
alt_fe_curve_d = np.zeros([curve_nsteps,n_curves])
alt_fe_curve_a = np.zeros([curve_nsteps,n_curves])
alt_fe_curve_b = np.zeros([curve_nsteps,n_curves])

dpri_curve = np.zeros([curve_nsteps,n_curves])
dpri_curve_d = np.zeros([curve_nsteps,n_curves])
dpri_curve_a = np.zeros([curve_nsteps,n_curves])
dpri_curve_b = np.zeros([curve_nsteps,n_curves])

dsec_curve = np.zeros([curve_nsteps,n_curves,minNum+1])
dsec_curve_d = np.zeros([curve_nsteps,n_curves,minNum+1])
dsec_curve_a = np.zeros([curve_nsteps,n_curves,minNum+1])
dsec_curve_b = np.zeros([curve_nsteps,n_curves,minNum+1])

secondary = np.array(['', 'kaolinite', 'saponite_mg', 'celadonite', 'clinoptilolite', 'pyrite', 'mont_na', 'goethite',
'smectite', 'calcite', 'kspar', 'saponite_na', 'nont_na', 'nont_mg', 'fe_celad', 'nont_ca',
'mesolite', 'hematite', 'mont_ca', 'verm_ca', 'analcime', 'philipsite', 'mont_mg', 'gismondine',
'verm_mg', 'natrolite', 'talc', 'smectite_low', 'prehnite', 'chlorite', 'scolecite', 'clinochlorte14a',
'clinochlore7a', 'saponite_ca', 'verm_na', 'pyrrhotite', 'fe_saponite_ca', 'fe_saponite_mg', 'daphnite7a', 'daphnite14a', 'epidote'])


#hack: alteration index data
nsites = 9
ebw = 800.0
dark_red = '#aaaaaa'
data_lw = 1
plot_purple = '#b678f5'
plot_blue = '#4e94c1'
lower_color = '#339a28'
upper_color = '#17599e'
fill_color = '#e4e4e4'

#poop: site locations
site_locations = np.array([22.742, 25.883, 33.872, 40.706, 45.633, 55.765, 75.368, 99.006, 102.491])
site_locations = (site_locations - 00.00)*1000.0

site_names = ["1023", "1024", "1025", "1031", "1028", "1029", "1032", "1026", "1027"]
alt_values = np.array([0.3219, 2.1072, 2.3626, 2.9470, 10.0476, 4.2820, 8.9219, 11.8331, 13.2392])
lower_eb = np.array([0.3219, 0.04506, 0.8783, 1.7094, 5.0974, 0.8994, 5.3745, 2.5097, 3.0084])
upper_eb = np.array([1.7081, 2.9330, 3.7662, 4.9273, 11.5331, 5.0247, 10.7375, 17.8566, 27.4308])

fe_values = np.array([0.7753, 0.7442, 0.7519, 0.7610, 0.6714, 0.7416, 0.7039, 0.6708, 0.6403])
lower_eb_fe = np.array([0.7753, 0.7442, 0.7208, 0.7409, 0.6240, 0.7260, 0.6584, 0.6299, 0.6084])
upper_eb_fe = np.array([0.7753, 0.7442, 0.7519, 0.7812, 0.7110, 0.7610, 0.7396, 0.7104, 0.7026])

# site_locations_1028
site_locations_1028 = np.array([22.742, 25.883, 33.872, 40.706, 55.765, 75.368, 99.006, 102.491])
site_locations_1028 = (site_locations_1028 - 00.00)*1000.0

alt_values_1028 = np.array([0.3219, 2.1072, 2.3626, 2.9470, 4.2820, 8.9219, 11.8331, 13.2392])

fe_values_1028 = np.array([0.7753, 0.7442, 0.7519, 0.7610, 0.7416, 0.7039, 0.6708, 0.6403])




#hack: lateral grid
n_lateral = 100
age_vector = np.linspace(0.8,3.5,n_lateral,endpoint=True)
distance_vector = np.linspace(0.0, 120000.0,n_lateral, endpoint=True)


compound_alt_vol_solo = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_solo = np.zeros([n_lateral, len(param_strings),n_grids])

compound_alt_vol_solo_shift = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_solo_shift = np.zeros([n_lateral, len(param_strings),n_grids])





compound_alt_vol_dual_max = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_dual_max = np.zeros([n_lateral, len(param_strings),n_grids])

compound_alt_vol_dual_max_shift = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_dual_max_shift = np.zeros([n_lateral, len(param_strings),n_grids])

compound_alt_vol_dual_min = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_dual_min = np.zeros([n_lateral, len(param_strings),n_grids])

compound_alt_vol_dual_min_shift = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_dual_min_shift = np.zeros([n_lateral, len(param_strings),n_grids])



# new a , b compounds
compound_alt_vol_a_max = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_a_max = np.zeros([n_lateral, len(param_strings),n_grids])

compound_alt_vol_a_max_shift = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_a_max_shift = np.zeros([n_lateral, len(param_strings),n_grids])

compound_alt_vol_a_min = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_a_min = np.zeros([n_lateral, len(param_strings),n_grids])

compound_alt_vol_a_min_shift = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_a_min_shift = np.zeros([n_lateral, len(param_strings),n_grids])




compound_alt_vol_b_max = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_b_max = np.zeros([n_lateral, len(param_strings),n_grids])

compound_alt_vol_b_max_shift = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_b_max_shift = np.zeros([n_lateral, len(param_strings),n_grids])

compound_alt_vol_b_min = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_b_min = np.zeros([n_lateral, len(param_strings),n_grids])

compound_alt_vol_b_min_shift = np.zeros([n_lateral, len(param_strings),n_grids])
compound_alt_fe_b_min_shift = np.zeros([n_lateral, len(param_strings),n_grids])




for iii in range(len(temp_string_list)):

    dir_path = "z_h_h_"+temp_string_list[iii]+"/"
    fig_path = "fig_lateral/"




    #todo: LOAD IN 2d alt index grids
    value_alt_vol_mean[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean.txt')
    value_alt_vol_mean_d[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean_d.txt')
    value_alt_vol_mean_a[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean_a.txt')
    value_alt_vol_mean_b[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean_b.txt')

    value_alt_fe_mean[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean.txt')
    value_alt_fe_mean_d[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean_d.txt')
    value_alt_fe_mean_a[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean_a.txt')
    value_alt_fe_mean_b[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean_b.txt')

    value_dpri_mean[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_dpri_mean.txt')
    value_dpri_mean_d[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_dpri_mean_d.txt')
    value_dpri_mean_a[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_dpri_mean_a.txt')
    value_dpri_mean_b[:,:,iii] = np.loadtxt(in_path + dir_path + 'value_dpri_mean_b.txt')

    #hack: load in SAVE JUNE
    save_feot_mean[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_feot_mean.txt')
    save_feot_mean_d[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_feot_mean_d.txt')
    save_feot_mean_a[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_feot_mean_a.txt')
    save_feot_mean_b[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_feot_mean_b.txt')

    save_mgo_mean[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_mgo_mean.txt')
    save_mgo_mean_d[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_mgo_mean_d.txt')
    save_mgo_mean_a[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_mgo_mean_a.txt')
    save_mgo_mean_b[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_mgo_mean_b.txt')

    save_k2o_mean[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_k2o_mean.txt')
    save_k2o_mean_d[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_k2o_mean_d.txt')
    save_k2o_mean_a[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_k2o_mean_a.txt')
    save_k2o_mean_b[:,:,iii] = np.loadtxt(in_path + dir_path + 'save_k2o_mean_b.txt')


    #todo: LOAD IN value_sec_x.txt

    any_min = []
    for j in range(1,minNum):
        if os.path.isfile(in_path + dir_path + 'value_dsec_'+str(int(j))+'.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            value_dsec[:,:,j,iii] = np.loadtxt(in_path + dir_path + 'value_dsec_'+str(int(j))+'.txt')

        if os.path.isfile(in_path + dir_path + 'value_dsec_'+str(int(j))+'_d.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            value_dsec_d[:,:,j,iii] = np.loadtxt(in_path + dir_path + 'value_dsec_'+str(int(j))+'_d.txt')

        if os.path.isfile(in_path + dir_path + 'value_dsec_'+str(int(j))+'_a.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            value_dsec_a[:,:,j,iii] = np.loadtxt(in_path + dir_path + 'value_dsec_'+str(int(j))+'_a.txt')

        if os.path.isfile(in_path + dir_path + 'value_dsec_'+str(int(j))+'_b.txt'):
            if not np.any(any_min == j):
                any_min = np.append(any_min,j)
            value_dsec_b[:,:,j,iii] = np.loadtxt(in_path + dir_path + 'value_dsec_'+str(int(j))+'_b.txt')

    print "any_min: " , any_min
    for ee in range(len(any_min)):
        print any_min[ee], secondary[any_min[ee]]


    for i in range(n_lateral-1):


        #hack: define the shift!!
        shift_myr = 0.0
        #poop: block_scale
        block_scale = 0.856#*(3.5/2.7)
        active_myr = 4.5*block_scale
        for ii in range(len(param_strings)):
            compound_alt_vol_solo[i,ii,iii] = (active_myr/n_lateral)*i*value_alt_vol_mean[ii,0,iii]
            compound_alt_fe_solo[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*value_alt_fe_mean[ii,0,iii]

            compound_alt_vol_dual_max[i,ii,iii] = (active_myr/n_lateral)*i*np.max(value_alt_vol_mean_d[:,:,iii])
            compound_alt_fe_dual_max[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.max(value_alt_fe_mean_d[:,:,iii])

            compound_alt_vol_dual_min[i,ii,iii] = (active_myr/n_lateral)*i*np.min(value_alt_vol_mean_d[:,:,iii])
            compound_alt_fe_dual_min[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.min(value_alt_fe_mean_d[:,:,iii])


            # new a
            compound_alt_vol_a_max[i,ii,iii] = (active_myr/n_lateral)*i*np.max(value_alt_vol_mean_a[:,:,iii])
            compound_alt_fe_a_max[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.max(value_alt_fe_mean_a[:,:,iii])
            compound_alt_vol_a_min[i,ii,iii] = (active_myr/n_lateral)*i*np.min(value_alt_vol_mean_a[:,:,iii])
            compound_alt_fe_a_min[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.min(value_alt_fe_mean_a[:,:,iii])

            # new b
            compound_alt_vol_b_max[i,ii,iii] = (active_myr/n_lateral)*i*np.max(value_alt_vol_mean_b[:,:,iii])
            compound_alt_fe_b_max[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.max(value_alt_fe_mean_b[:,:,iii])
            compound_alt_vol_b_min[i,ii,iii] = (active_myr/n_lateral)*i*np.min(value_alt_vol_mean_b[:,:,iii])
            compound_alt_fe_b_min[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.min(value_alt_fe_mean_b[:,:,iii])

            if age_vector[i] > shift_myr:
                compound_alt_vol_solo_shift[i,ii,iii] = (active_myr/n_lateral)*i*value_alt_vol_mean[ii,0,iii] - shift_myr*value_alt_vol_mean[ii,0,iii]
                compound_alt_fe_solo_shift[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*value_alt_fe_mean[ii,0,iii] + shift_myr*value_alt_fe_mean[ii,0,iii]

                compound_alt_vol_dual_max_shift[i,ii,iii] = (active_myr/n_lateral)*i*np.max(value_alt_vol_mean_d[:,:,iii]) - shift_myr*np.max(value_alt_vol_mean_d[:,:,iii])
                compound_alt_fe_dual_max_shift[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.max(value_alt_fe_mean_d[:,:,iii]) + shift_myr*np.max(value_alt_fe_mean_d[:,:,iii])

                compound_alt_vol_dual_min_shift[i,ii,iii] = (active_myr/n_lateral)*i*np.min(value_alt_vol_mean_d[:,:,iii]) - shift_myr*np.min(value_alt_vol_mean_d[:,:,iii])
                compound_alt_fe_dual_min_shift[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.min(value_alt_fe_mean_d[:,:,iii]) + shift_myr*np.min(value_alt_fe_mean_d[:,:,iii])

                # new a
                compound_alt_vol_a_max_shift[i,ii,iii] = (active_myr/n_lateral)*i*np.max(value_alt_vol_mean_a[:,:,iii]) - shift_myr*np.max(value_alt_vol_mean_a[:,:,iii])
                compound_alt_fe_a_max_shift[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.max(value_alt_fe_mean_a[:,:,iii]) + shift_myr*np.max(value_alt_fe_mean_a[:,:,iii])

                compound_alt_vol_a_min_shift[i,ii,iii] = (active_myr/n_lateral)*i*np.min(value_alt_vol_mean_a[:,:,iii]) - shift_myr*np.min(value_alt_vol_mean_a[:,:,iii])
                compound_alt_fe_a_min_shift[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.min(value_alt_fe_mean_a[:,:,iii]) + shift_myr*np.min(value_alt_fe_mean_a[:,:,iii])

                # new b
                compound_alt_vol_b_max_shift[i,ii,iii] = (active_myr/n_lateral)*i*np.max(value_alt_vol_mean_b[:,:,iii]) - shift_myr*np.max(value_alt_vol_mean_b[:,:,iii])
                compound_alt_fe_b_max_shift[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.max(value_alt_fe_mean_b[:,:,iii]) + shift_myr*np.max(value_alt_fe_mean_b[:,:,iii])

                compound_alt_vol_b_min_shift[i,ii,iii] = (active_myr/n_lateral)*i*np.min(value_alt_vol_mean_b[:,:,iii]) - shift_myr*np.min(value_alt_vol_mean_b[:,:,iii])
                compound_alt_fe_b_min_shift[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.min(value_alt_fe_mean_b[:,:,iii]) + shift_myr*np.min(value_alt_fe_mean_b[:,:,iii])



    #poop: recalibrate
    if iii == len(temp_string_list)-2:
        print "len(temp_string_list)-2 " , temp_string_list[iii]
        print "a max slope: " , np.max(value_alt_vol_mean_a[:,:,iii])# * (active_myr/n_lateral)
        print "a min slope: " , np.min(value_alt_vol_mean_a[:,:,iii])# * (active_myr/n_lateral)

        print "len(temp_string_list)-2 " , temp_string_list[iii]
        print "b max slope: " , np.max(value_alt_vol_mean_b[:,:,iii])# * (active_myr/n_lateral)
        print "b min slope: " , np.min(value_alt_vol_mean_b[:,:,iii])# * (active_myr/n_lateral)

        print " "
        print " "

        print "len(temp_string_list)-2 " , temp_string_list[iii]
        print "a max slope fe: " , np.max(value_alt_fe_mean_a[:,:,iii])# * (active_myr/n_lateral)
        print "a min slope fe: " , np.min(value_alt_fe_mean_a[:,:,iii])# * (active_myr/n_lateral)

        print "len(temp_string_list)-2 " , temp_string_list[iii]
        print "b max slope fe: " , np.max(value_alt_fe_mean_b[:,:,iii])# * (active_myr/n_lateral)
        print "b min slope fe: " , np.min(value_alt_fe_mean_b[:,:,iii])# * (active_myr/n_lateral)










#hack: FIG: comp_lin_solo_36
print "comp_lin_solo_36"
fig=plt.figure(figsize=(16.0,8.0))
rainbow_lw = 1.0

ax=fig.add_subplot(2, 3, 1, frameon=True)


for ii in range(len(param_strings)):
    plt.plot(distance_vector,compound_alt_vol_solo[:,ii], label=str(param_strings[ii]), c=plot_col[ii+1], lw=rainbow_lw, zorder=10)

plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

plt.xlim([20000.0,110000.0])
plt.ylim([-1.0,30.0])
# plt.xlabel("crust age [Myr]", fontsize=9)
plt.ylabel("alteration volume percent", fontsize=9)
#plt.legend(fontsize=8,loc='best',ncol=1)





ax=fig.add_subplot(2, 3, 2, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(distance_vector,compound_alt_vol_solo_shift[:,ii], label=str(param_strings[ii]), c=plot_col[ii+1], lw=rainbow_lw, zorder=10)

plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

plt.xlim([20000.0,110000.0])
plt.ylim([-1.0,30.0])
# plt.xlabel("crust age [Myr]", fontsize=9)
plt.ylabel("alteration volume percent", fontsize=9)
#plt.legend(fontsize=8,loc='best',ncol=1)








ax=fig.add_subplot(2, 3, 4, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(distance_vector,compound_alt_fe_solo[:,ii], label=str(param_strings[ii]), c=plot_col[ii+1], lw=rainbow_lw, zorder=10)

plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

plt.xlim([20000.0,110000.0])
plt.ylim([0.6,0.8])
plt.ylabel("FeO/FeOt", fontsize=9)
#plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))





ax=fig.add_subplot(2, 3, 5, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(distance_vector,compound_alt_fe_solo_shift[:,ii], label=str(param_strings[ii]), c=plot_col[ii+1], lw=rainbow_lw, zorder=10)

plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

plt.xlim([20000.0,110000.0])
plt.ylim([0.6,0.8])
plt.ylabel("FeO/FeOt", fontsize=9)
#plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))




plt.savefig(brk_path+"z_comp_lin_solo_36.png",bbox_inches='tight')
















dual_alpha = 0.4
hatch_alpha = 1.0
hatch_string = '||'
hatch_strings = ['\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\']
#hatch_strings = ['\\\\', 'o', '//', '++', '..']
hatch_strings = ['\\\\', '.', '\\\\', '||', '\\\\', '.', '\\\\', '||']
solo_alpha = 0.4
lw_for_plot = 2.0

#hack: FIG: comp_lin_fill_36
print "comp_lin_fill_36"
fig=plt.figure(figsize=(16.0,8.0))
rainbow_lw = 1.0

ax=fig.add_subplot(2, 3, 1, frameon=True)

for iii in range(len(temp_string_list)):
    # ax.fill_between(distance_vector, compound_alt_vol_solo[:,0,iii], compound_alt_vol_solo[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
    ax.fill_between(distance_vector, compound_alt_vol_dual_max[:,0,iii], compound_alt_vol_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])


    # plt.plot(distance_vector, compound_alt_vol_dual_max[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
    # plt.plot(distance_vector, compound_alt_vol_dual_min[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])

    plt.plot(distance_vector, compound_alt_vol_solo[:,:,iii], lw=lw_for_plot, color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])


plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([-1.0,30.0])
# plt.xlabel("crust age [Myr]", fontsize=9)
plt.ylabel("alteration volume percent", fontsize=9)
#plt.legend(fontsize=8,loc='best',ncol=1)





ax=fig.add_subplot(2, 3, 2, frameon=True)


for iii in range(len(temp_string_list)):
    # ax.fill_between(distance_vector, compound_alt_vol_solo_shift[:,0,iii], compound_alt_vol_solo_shift[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
    jump = ax.fill_between(distance_vector, compound_alt_vol_dual_max_shift[:,0,iii], compound_alt_vol_dual_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])

    # plt.plot(distance_vector, compound_alt_vol_dual_max_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
    # plt.plot(distance_vector, compound_alt_vol_dual_min_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])

    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,:,iii], lw=lw_for_plot, color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])



plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([-1.0,30.0])
# plt.xlabel("crust age [Myr]", fontsize=9)
plt.ylabel("alteration volume percent", fontsize=9)
#plt.legend(fontsize=8,loc='best',ncol=1)














hatch_strings = ['//', '//', '//', '//', '//','//', '//', '//', '//', '//']



ax=fig.add_subplot(2, 3, 4, frameon=True)


for iii in range(len(temp_string_list)):
    # ax.fill_between(distance_vector, compound_alt_fe_solo[:,0,iii], compound_alt_fe_solo[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
    ax.fill_between(distance_vector, compound_alt_fe_dual_max[:,0,iii], compound_alt_fe_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])

    # plt.plot(distance_vector, compound_alt_fe_dual_max[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
    # plt.plot(distance_vector, compound_alt_fe_dual_min[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])

    plt.plot(distance_vector, compound_alt_fe_solo[:,0,iii], lw=lw_for_plot, color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_fe_solo[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])

plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.6,0.8])
plt.ylabel("FeO/FeOt", fontsize=9)
#plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))





ax=fig.add_subplot(2, 3, 5, frameon=True)

for iii in range(len(temp_string_list)):
    # ax.fill_between(distance_vector, compound_alt_fe_solo_shift[:,0,iii], compound_alt_fe_solo_shift[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
    ax.fill_between(distance_vector, compound_alt_fe_dual_max_shift[:,0,iii], compound_alt_fe_dual_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])

    # plt.plot(distance_vector, compound_alt_fe_dual_max_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
    # plt.plot(distance_vector, compound_alt_fe_dual_min_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])

    plt.plot(distance_vector, compound_alt_fe_solo_shift[:,0,iii], lw=lw_for_plot, color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_fe_solo_shift[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])

plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.6,0.8])
plt.ylabel("FeO/FeOt", fontsize=9)
#plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))




plt.savefig(brk_path+"z_comp_lin_fill_36.png",bbox_inches='tight')




#poop: confidence interval
#st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))

print " "

slope, intercept, r_value, p_value, std_err = stats.linregress(site_locations,alt_values)
print "slope: " , slope
print "intercept: " , intercept
print "r_value: " , r_value
print "p_value: " , p_value
print "std_err: " , std_err

print " "



slope_fe, intercept_fe, r_value_fe, p_value_fe, std_err_fe = stats.linregress(site_locations,fe_values)
print "slope_fe: " , slope_fe
print "intercept_fe: " , intercept_fe
print "r_value_fe: " , r_value_fe
print "p_value_fe: " , p_value_fe
print "std_err_fe: " , std_err_fe

print " "



slope_1028, intercept_1028, r_value_1028, p_value_1028, std_err_1028 = stats.linregress(site_locations_1028,alt_values_1028)
print "slope_1028: " , slope_1028
print "intercept_1028: " , intercept_1028
print "r_value_1028: " , r_value_1028
print "p_value_1028: " , p_value_1028
print "std_err_1028: " , std_err_1028

print " "



slope_fe_1028, intercept_fe_1028, r_value_fe_1028, p_value_fe_1028, std_err_fe_1028 = stats.linregress(site_locations_1028,fe_values_1028)
print "slope_fe_1028: " , slope_fe_1028
print "intercept_fe_1028: " , intercept_fe_1028
print "r_value_fe_1028: " , r_value_fe_1028
print "p_value_fe_1028: " , p_value_fe_1028
print "std_err_fe_1028: " , std_err_fe_1028

print " "



dual_alpha = 0.4
hatch_alpha = 1.0
hatch_string = '||'
hatch_strings = ['\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\']
#hatch_strings = ['\\\\', 'o', '//', '++', '..']
hatch_strings = ['\\\\', '.', '\\\\', '||', '\\\\', '.', '\\\\', '||']
solo_alpha = 0.4
lw_for_plot = 1.0

#hack: FIG: y_long_trial_36
print "y_long_trial_36"
fig=plt.figure(figsize=(16.0,8.0))
rainbow_lw = 1.0

dark_red = '#373c41'



ax=fig.add_subplot(2, 3, 1, frameon=True)

# for iii in range(len(temp_string_list)):

    # if iii == 0 or iii == 2 or iii == 4 or iii == 6:
    #     # ax.fill_between(distance_vector, compound_alt_vol_solo[:,0,iii], compound_alt_vol_solo[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
    #     ax.fill_between(distance_vector, compound_alt_vol_dual_max[:,0,iii], compound_alt_vol_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])
    #
    #
    #     # plt.plot(distance_vector, compound_alt_vol_dual_max[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
    #     # plt.plot(distance_vector, compound_alt_vol_dual_min[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
    #
    #     plt.plot(distance_vector, compound_alt_vol_solo[:,:,iii], lw=lw_for_plot, color=plot_col[iii+1])
    #     #plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])


plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

#poop: alt_vol regressions
the_slope, the_intercept = np.polyfit(site_locations, alt_values, 1)
the_slope_1028, the_intercept_1028 = np.polyfit(site_locations_1028, alt_values_1028, 1)
plt.plot(site_locations, site_locations*the_slope + the_intercept, lw=2.0, color='k')
plt.plot(site_locations,site_locations*(the_slope+(2.64e-5)+(2.64e-5)) + the_intercept, lw=1.0, color='k')
plt.plot(site_locations,site_locations*(the_slope-(2.64e-5)-(2.64e-5)) + the_intercept, lw=1.0, color='k')


plt.plot(site_locations_1028, site_locations_1028*the_slope_1028 + the_intercept_1028, lw=2.0, color='m')
plt.plot(site_locations_1028,site_locations_1028*(the_slope_1028+(9.13e-6)+(9.13e-6)) + the_intercept_1028, lw=1.0, color='m')
plt.plot(site_locations_1028,site_locations_1028*(the_slope_1028-(9.13e-6)-(9.13e-6)) + the_intercept_1028, lw=1.0, color='m')


# plt.plot(site_locations, site_locations*0.00012601 + the_intercept, lw=2.0, color='m')
# plt.plot(site_locations, site_locations*0.00015999 + the_intercept, lw=2.0, color='m')
#plt.plot([distance_vector[0],distance_vector[-1]])
print "the_slope: " , the_slope
print "the_intercept: " , the_intercept


plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([-1.0,30.0])
# plt.xlabel("crust age [Myr]", fontsize=9)
plt.ylabel("alteration volume percent", fontsize=9)
#plt.legend(fontsize=8,loc='best',ncol=1)





ax=fig.add_subplot(2, 3, 2, frameon=True)


for iii in range(len(temp_string_list)):

    if iii == 0 or iii == 2 or iii == 4 or iii == 6:
        # ax.fill_between(distance_vector, compound_alt_vol_solo_shift[:,0,iii], compound_alt_vol_solo_shift[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
        jump = ax.fill_between(distance_vector, compound_alt_vol_dual_max_shift[:,0,iii], compound_alt_vol_dual_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])

        # plt.plot(distance_vector, compound_alt_vol_dual_max_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
        # plt.plot(distance_vector, compound_alt_vol_dual_min_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])

        plt.plot(distance_vector, compound_alt_vol_solo_shift[:,:,iii], lw=lw_for_plot, color=plot_col[iii+1])
        #plt.plot(distance_vector, compound_alt_vol_solo_shift[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])

plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)



plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([-1.0,30.0])
# plt.xlabel("crust age [Myr]", fontsize=9)
plt.ylabel("alteration volume percent", fontsize=9)
#plt.legend(fontsize=8,loc='best',ncol=1)




hatch_strings = ['//', '//', '//', '//', '//','//', '//', '//', '//', '//']


ax=fig.add_subplot(2, 3, 4, frameon=True)


# for iii in range(len(temp_string_list)):

    # if iii == 0 or iii == 2 or iii == 4 or iii == 6:
    #
    #     # ax.fill_between(distance_vector, compound_alt_fe_solo[:,0,iii], compound_alt_fe_solo[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
    #     ax.fill_between(distance_vector, compound_alt_fe_dual_max[:,0,iii], compound_alt_fe_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])
    #
    #     # plt.plot(distance_vector, compound_alt_fe_dual_max[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
    #     # plt.plot(distance_vector, compound_alt_fe_dual_min[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
    #
    #     plt.plot(distance_vector, compound_alt_fe_solo[:,:,iii], lw=lw_for_plot, color=plot_col[iii+1])
    #     #plt.plot(distance_vector, compound_alt_fe_solo[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])

plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

the_slope_fe, the_intercept_fe = np.polyfit(site_locations, fe_values, 1)
the_slope_fe_1028, the_intercept_fe_1028 = np.polyfit(site_locations_1028, fe_values_1028, 1)

#poop: fe regressions
plt.plot(site_locations, site_locations*the_slope_fe + the_intercept_fe, lw=2.0, color='k')
plt.plot(site_locations,site_locations*(the_slope_fe+(3.32e-7)+(3.32e-7)) + the_intercept_fe, lw=1.0, color='k')
plt.plot(site_locations,site_locations*(the_slope_fe-(3.32e-7)-(3.32e-7)) + the_intercept_fe, lw=1.0, color='k')


plt.plot(site_locations_1028, site_locations_1028*the_slope_fe_1028 + the_intercept_fe_1028, lw=2.0, color='m')
plt.plot(site_locations_1028,site_locations_1028*(the_slope_fe_1028+(1.87e-7)+(1.87e-7)) + the_intercept_fe_1028, lw=1.0, color='m')
plt.plot(site_locations_1028,site_locations_1028*(the_slope_fe_1028-(1.87e-7)-(1.87e-7)) + the_intercept_fe_1028, lw=1.0, color='m')

print "the_slope_fe: " , the_slope_fe
print "the_intercept_fe: " , the_intercept_fe

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.6,0.8])
plt.ylabel("FeO/FeOt", fontsize=9)
#plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))





ax=fig.add_subplot(2, 3, 5, frameon=True)

for iii in range(len(temp_string_list)):

    if iii == 0 or iii == 2 or iii == 4 or iii == 6:

        # ax.fill_between(distance_vector, compound_alt_fe_solo_shift[:,0,iii], compound_alt_fe_solo_shift[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
        ax.fill_between(distance_vector, compound_alt_fe_dual_max_shift[:,0,iii], compound_alt_fe_dual_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])

        # plt.plot(distance_vector, compound_alt_fe_dual_max_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
        # plt.plot(distance_vector, compound_alt_fe_dual_min_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])

        plt.plot(distance_vector, compound_alt_fe_solo_shift[:,:,iii], lw=lw_for_plot, color=plot_col[iii+1])
        #plt.plot(distance_vector, compound_alt_fe_solo_shift[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])

plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.6,0.8])
plt.ylabel("FeO/FeOt", fontsize=9)
#plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))




plt.savefig(brk_path+"y_long_trial_36.png",bbox_inches='tight')















dual_alpha = 0.4
hatch_alpha = 1.0
hatch_string = '||'
hatch_strings = ['\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\']
#hatch_strings = ['\\\\', 'o', '//', '++', '..']
hatch_strings = ['\\\\', '.', '\\\\', '||', '\\\\', '.', '\\\\', '||']
solo_alpha = 0.4
lw_for_plot = 1.0

#hack: FIG: yy_long_trial_36
print "yy_long_trial_36"
fig=plt.figure(figsize=(16.0,8.0))
rainbow_lw = 1.0



ax=fig.add_subplot(2, 3, 1, frameon=True)

for iii in range(len(temp_string_list)):

    if iii == 1 or iii == 3 or iii == 5 or iii == 7:
        # ax.fill_between(distance_vector, compound_alt_vol_solo[:,0,iii], compound_alt_vol_solo[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
        ax.fill_between(distance_vector, compound_alt_vol_dual_max[:,0,iii], compound_alt_vol_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])


        # plt.plot(distance_vector, compound_alt_vol_dual_max[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
        # plt.plot(distance_vector, compound_alt_vol_dual_min[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])

        plt.plot(distance_vector, compound_alt_vol_solo[:,:,iii], lw=lw_for_plot, color=plot_col[iii+1])
        #plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])


plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([-1.0,30.0])
# plt.xlabel("crust age [Myr]", fontsize=9)
plt.ylabel("alteration volume percent", fontsize=9)
#plt.legend(fontsize=8,loc='best',ncol=1)





ax=fig.add_subplot(2, 3, 2, frameon=True)


for iii in range(len(temp_string_list)):

    if iii == 1 or iii == 3 or iii == 5 or iii == 7:
        # ax.fill_between(distance_vector, compound_alt_vol_solo_shift[:,0,iii], compound_alt_vol_solo_shift[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
        jump = ax.fill_between(distance_vector, compound_alt_vol_dual_max_shift[:,0,iii], compound_alt_vol_dual_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])

        # plt.plot(distance_vector, compound_alt_vol_dual_max_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
        # plt.plot(distance_vector, compound_alt_vol_dual_min_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])

        plt.plot(distance_vector, compound_alt_vol_solo_shift[:,:,iii], lw=lw_for_plot, color=plot_col[iii+1])
        #plt.plot(distance_vector, compound_alt_vol_solo_shift[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])

plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([-1.0,30.0])
# plt.xlabel("crust age [Myr]", fontsize=9)
plt.ylabel("alteration volume percent", fontsize=9)
#plt.legend(fontsize=8,loc='best',ncol=1)




hatch_strings = ['//', '//', '//', '//', '//','//', '//', '//', '//', '//']


ax=fig.add_subplot(2, 3, 4, frameon=True)


for iii in range(len(temp_string_list)):

    if iii == 1 or iii == 3 or iii == 5 or iii == 7:

        # ax.fill_between(distance_vector, compound_alt_fe_solo[:,0,iii], compound_alt_fe_solo[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
        ax.fill_between(distance_vector, compound_alt_fe_dual_max[:,0,iii], compound_alt_fe_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])

        # plt.plot(distance_vector, compound_alt_fe_dual_max[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
        # plt.plot(distance_vector, compound_alt_fe_dual_min[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])

        plt.plot(distance_vector, compound_alt_fe_solo[:,:,iii], lw=lw_for_plot, color=plot_col[iii+1])
        #plt.plot(distance_vector, compound_alt_fe_solo[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])

plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.6,0.8])
plt.ylabel("FeO/FeOt", fontsize=9)
#plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))





ax=fig.add_subplot(2, 3, 5, frameon=True)

for iii in range(len(temp_string_list)):

    if iii == 1 or iii == 3 or iii == 5 or iii == 7:

        # ax.fill_between(distance_vector, compound_alt_fe_solo_shift[:,0,iii], compound_alt_fe_solo_shift[:,-1,iii], facecolor=plot_col[iii+1], lw=0, zorder=15-iii, alpha=solo_alpha)
        ax.fill_between(distance_vector, compound_alt_fe_dual_max_shift[:,0,iii], compound_alt_fe_dual_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_strings[iii], edgecolor=plot_col_hatch[iii+1])

        # plt.plot(distance_vector, compound_alt_fe_dual_max_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])
        # plt.plot(distance_vector, compound_alt_fe_dual_min_shift[:,0,iii], lw=lw_for_plot, color=plot_col_hatch[iii+1])

        plt.plot(distance_vector, compound_alt_fe_solo_shift[:,:,iii], lw=lw_for_plot, color=plot_col[iii+1])
        #plt.plot(distance_vector, compound_alt_fe_solo_shift[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])

plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.6,0.8])
plt.ylabel("FeO/FeOt", fontsize=9)
#plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))




plt.savefig(brk_path+"yy_long_trial_36.png",bbox_inches='tight')




#todo: smart_bars

bar_min_s = np.zeros(len(temp_string_list))
bar_max_s = np.zeros(len(temp_string_list))
bar_min_a = np.zeros(len(temp_string_list))
bar_max_a = np.zeros(len(temp_string_list))
bar_min_b = np.zeros(len(temp_string_list))
bar_max_b = np.zeros(len(temp_string_list))

bar_fe_min_s = np.zeros(len(temp_string_list))
bar_fe_max_s = np.zeros(len(temp_string_list))
bar_fe_min_a = np.zeros(len(temp_string_list))
bar_fe_max_a = np.zeros(len(temp_string_list))
bar_fe_min_b = np.zeros(len(temp_string_list))
bar_fe_max_b = np.zeros(len(temp_string_list))

value_alt_fe_mean = np.abs(value_alt_fe_mean)
value_alt_fe_mean_d = np.abs(value_alt_fe_mean_d)
value_alt_fe_mean_a = np.abs(value_alt_fe_mean_a)
value_alt_fe_mean_b = np.abs(value_alt_fe_mean_b)

nn = range(len(temp_string_list))
for iii in nn:
    bar_min_s[iii] = np.min(value_alt_vol_mean[:,:,iii])
    bar_max_s[iii] = np.max(value_alt_vol_mean[:,:,iii])

    bar_min_a[iii] = np.min(value_alt_vol_mean_a[:,:,iii])
    bar_max_a[iii] = np.max(value_alt_vol_mean_a[:,:,iii])

    bar_min_b[iii] = np.min(value_alt_vol_mean_b[:,:,iii])
    bar_max_b[iii] = np.max(value_alt_vol_mean_b[:,:,iii])


    bar_fe_min_s[iii] = np.min(value_alt_fe_mean[:,:,iii])
    bar_fe_max_s[iii] = np.max(value_alt_fe_mean[:,:,iii])

    bar_fe_min_a[iii] = np.min(value_alt_fe_mean_a[:,:,iii])
    bar_fe_max_a[iii] = np.max(value_alt_fe_mean_a[:,:,iii])

    bar_fe_min_b[iii] = np.min(value_alt_fe_mean_b[:,:,iii])
    bar_fe_max_b[iii] = np.max(value_alt_fe_mean_b[:,:,iii])

fig=plt.figure(figsize=(17.0,14.0))
plt.subplots_adjust(wspace=0.17, hspace=0.1)


ax=fig.add_subplot(2, 2, 1, frameon=True)

plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
plt.scatter(site_locations, alt_values, s=60, marker='s', c='#87bed0',zorder=22)
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

plt.plot([20000.0,site_locations[-1]] , [0.0, upper_eb[-1]],color='m', ls='--')
plt.plot([20000.0,site_locations[-2]] , [0.0, lower_eb[-2]],color='m', ls='--')

#poop: alt_vol regressions
the_slope, the_intercept = np.polyfit(site_locations, alt_values, 1)
the_slope_1028, the_intercept_1028 = np.polyfit(site_locations_1028, alt_values_1028, 1)
plt.plot(site_locations, site_locations*the_slope + the_intercept, lw=2.0, color='k')
plt.plot(site_locations,site_locations*(the_slope+(2.64e-5)+(2.64e-5)) + the_intercept, lw=1.0, color='k')
plt.plot(site_locations,site_locations*(the_slope-(2.64e-5)-(2.64e-5)) + the_intercept, lw=1.0, color='k')


plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(20,110,10))
plt.ylim([-1.0,30.0])
plt.xlabel('distance from ridge axis [km]')
plt.ylabel('secondary volume [percent]')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(11)



ax=fig.add_subplot(2, 2, 3, frameon=True)

plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
plt.scatter(site_locations, fe_values, s=60, marker='s', c='#87bed0',zorder=22)
for j in range(nsites):
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

plt.plot([20000.0, site_locations[-3]] , [0.78, upper_eb_fe[-3]],color='m', ls='--')
plt.plot([20000.0, site_locations[-1]] , [0.78, lower_eb_fe[-1]],color='m', ls='--')

the_slope_fe, the_intercept_fe = np.polyfit(site_locations, fe_values, 1)
the_slope_fe_1028, the_intercept_fe_1028 = np.polyfit(site_locations_1028, fe_values_1028, 1)
#poop: fe regressions
plt.plot(site_locations, site_locations*the_slope_fe + the_intercept_fe, lw=2.0, color='k')
plt.plot(site_locations,site_locations*(the_slope_fe+(3.32e-7)+(3.32e-7)) + the_intercept_fe, lw=1.0, color='k')
plt.plot(site_locations,site_locations*(the_slope_fe-(3.32e-7)-(3.32e-7)) + the_intercept_fe, lw=1.0, color='k')


plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(20,110,10))
plt.ylim([0.6,0.8])
plt.xlabel('distance from ridge axis [km]')
plt.ylabel('FeO/FeOt')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(11)




ax=fig.add_subplot(2, 2, 2, frameon=True)

x_wee = 0.5
x_shft = 1.5

for iii in nn:
    ax.fill_between([temp_int_list[iii]-x_wee-x_shft,temp_int_list[iii]+x_wee-x_shft],[bar_min_s[iii],bar_min_s[iii]],[bar_max_s[iii],bar_max_s[iii]],
    facecolor='#b4afaf')

    ax.fill_between([temp_int_list[iii]-x_wee,temp_int_list[iii]+x_wee],[bar_min_a[iii],bar_min_a[iii]],[bar_max_a[iii],bar_max_a[iii]],
    facecolor='#c48080')

    ax.fill_between([temp_int_list[iii]-x_wee+x_shft,temp_int_list[iii]+x_wee+x_shft],[bar_min_b[iii],bar_min_b[iii]],[bar_max_b[iii],bar_max_b[iii]],
    facecolor='#477b85')

plt.plot([25.0,70.0], [1.11, 1.11], color='m', ls='--')
plt.plot([25.0,70.0], [10.28, 10.28], color='m', ls='--')

plt.xlim([25.0, 70.0])
plt.xlabel('temperature [C]')
plt.ylabel('range of secondary volume growth rates')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(11)



ax=fig.add_subplot(2, 2, 4, frameon=True)

x_wee = 0.5
x_shft = 1.5

for iii in nn:
    ax.fill_between([temp_int_list[iii]-x_wee-x_shft,temp_int_list[iii]+x_wee-x_shft],[bar_fe_min_s[iii],bar_fe_min_s[iii]],[bar_fe_max_s[iii],bar_fe_max_s[iii]],
    facecolor='#b4afaf')

    ax.fill_between([temp_int_list[iii]-x_wee,temp_int_list[iii]+x_wee],[bar_fe_min_a[iii],bar_fe_min_a[iii]],[bar_fe_max_a[iii],bar_fe_max_a[iii]],
    facecolor='#c48080')

    ax.fill_between([temp_int_list[iii]-x_wee+x_shft,temp_int_list[iii]+x_wee+x_shft],[bar_fe_min_b[iii],bar_fe_min_b[iii]],[bar_fe_max_b[iii],bar_fe_max_b[iii]],
    facecolor='#477b85')

plt.plot([25.0,70.0], [0.0027778,0.0027778], color='m', ls='--')
plt.plot([25.0,70.0], [0.0630,0.0630], color='m', ls='--')

plt.xlim([25.0, 70.0])
plt.xlabel('temperature [C]')
plt.ylabel('range of FeO/FeOt rates')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(11)



plt.savefig(brk_path+"smart_bars.png",bbox_inches='tight')














dual_alpha = 0.4
hatch_alpha = 1.0
hatch_string = '||'
hatch_strings = ['\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\']
#hatch_strings = ['\\\\', 'o', '//', '++', '..']
hatch_strings = ['\\\\', '.', '\\\\', '||', '\\\\', '.', '\\\\', '||']
solo_alpha = 0.4
lw_for_plot = 1.0

hatch_a = '|||'
hatch_b = '..'

fill_color = '#f1f1f1'
new_grey = '#535353'

#hack: FIG: yyy_long_trial_36
print "yyy_long_trial_36"
fig=plt.figure(figsize=(32.0,7.5))
rainbow_lw = 1.0



nub = range(len(temp_string_list))

for iii in nub[2:]:

    ax=fig.add_subplot(2, len(temp_string_list), iii+1, frameon=True)


    # ax.fill_between(distance_vector, compound_alt_vol_dual_max[:,0,iii], compound_alt_vol_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_a, edgecolor=plot_col_hatch[iii+1])
    #
    # plt.plot(distance_vector, compound_alt_vol_solo[:,:,iii], lw=0.5, color=plot_col[iii+1])
    # #plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])


    jump = ax.fill_between(distance_vector, compound_alt_vol_dual_max_shift[:,0,iii], compound_alt_vol_dual_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_b, edgecolor=new_grey)

    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,:,iii], lw=0.5, color=new_grey)


    plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
    for j in range(nsites):
        plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

    plt.xlim([20000.0,110000.0])
    plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
    plt.ylim([-1.0,30.0])
    plt.title(temp_string_list[iii])
    # plt.xlabel("crust age [Myr]", fontsize=9)
    # plt.ylabel("alteration volume percent", fontsize=9)
    #plt.legend(fontsize=8,loc='best',ncol=1)








hatch_strings = ['//', '//', '//', '//', '//','//', '//', '//', '//', '//']


ax=fig.add_subplot(2, len(temp_string_list), len(temp_string_list) + iii+1, frameon=True)


for iii in nub[2:]:


    ax=fig.add_subplot(2, len(temp_string_list), len(temp_string_list) + iii+1, frameon=True)

    # ax.fill_between(distance_vector, compound_alt_fe_dual_max[:,0,iii], compound_alt_fe_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_a, edgecolor=plot_col_hatch[iii+1])
    #
    # plt.plot(distance_vector, compound_alt_fe_solo[:,:,iii], lw=0.5, color=plot_col[iii+1])


    ax.fill_between(distance_vector, compound_alt_fe_dual_max_shift[:,0,iii], compound_alt_fe_dual_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_b, edgecolor=new_grey)

    plt.plot(distance_vector, compound_alt_fe_solo_shift[:,:,iii], lw=0.5, color=new_grey)


    plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
    for j in range(nsites):
        plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
    ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

    plt.xlim([20000.0,110000.0])
    plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
    plt.ylim([0.6,0.8])
    # plt.ylabel("FeO/FeOt", fontsize=9)
    #plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))




plt.savefig(brk_path+"yyy_long_trial_36.png",bbox_inches='tight')












dual_alpha = 0.4
hatch_alpha = 1.0
hatch_string = '||'
hatch_strings = ['\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\']
#hatch_strings = ['\\\\', 'o', '//', '++', '..']
hatch_strings = ['\\\\', '.', '\\\\', '||', '\\\\', '.', '\\\\', '||']
solo_alpha = 0.4
lw_for_plot = 1.0

hatch_a = '|||'
hatch_b = '|||'

fill_color = '#f1f1f1'
new_grey = '#317783'

#hack: FIG: yyy_a_long_trial_36
print "yyy_a_long_trial_36"
fig=plt.figure(figsize=(32.0,7.5))
rainbow_lw = 1.0



nub = range(len(temp_string_list))


for iii in nub[2:]:

    ax=fig.add_subplot(2, len(temp_string_list), iii+1, frameon=True)


    # ax.fill_between(distance_vector, compound_alt_vol_dual_max[:,0,iii], compound_alt_vol_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_a, edgecolor=plot_col_hatch[iii+1])
    #
    # plt.plot(distance_vector, compound_alt_vol_solo[:,:,iii], lw=0.5, color=plot_col[iii+1])
    # #plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])


    jump = ax.fill_between(distance_vector, compound_alt_vol_a_max_shift[:,0,iii], compound_alt_vol_a_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_b, edgecolor=new_grey)

    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,:,iii], lw=0.5, color=new_grey)


    plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
    for j in range(nsites):
        plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

    plt.xlim([20000.0,110000.0])
    plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
    plt.ylim([-1.0,30.0])
    plt.title(temp_string_list[iii])
    # plt.xlabel("crust age [Myr]", fontsize=9)
    # plt.ylabel("alteration volume percent", fontsize=9)
    #plt.legend(fontsize=8,loc='best',ncol=1)








hatch_strings = ['//', '//', '//', '//', '//','//', '//', '//', '//', '//']


ax=fig.add_subplot(2, len(temp_string_list), len(temp_string_list) + iii+1, frameon=True)


for iii in nub[2:]:


    ax=fig.add_subplot(2, len(temp_string_list), len(temp_string_list) + iii+1, frameon=True)

    # ax.fill_between(distance_vector, compound_alt_fe_dual_max[:,0,iii], compound_alt_fe_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_a, edgecolor=plot_col_hatch[iii+1])
    #
    # plt.plot(distance_vector, compound_alt_fe_solo[:,:,iii], lw=0.5, color=plot_col[iii+1])


    ax.fill_between(distance_vector, compound_alt_fe_a_max_shift[:,0,iii], compound_alt_fe_a_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_b, edgecolor=new_grey)

    plt.plot(distance_vector, compound_alt_fe_solo_shift[:,:,iii], lw=0.5, color=new_grey)


    plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
    for j in range(nsites):
        plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
    ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

    plt.xlim([20000.0,110000.0])
    plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
    plt.ylim([0.6,0.8])
    # plt.ylabel("FeO/FeOt", fontsize=9)
    #plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))




plt.savefig(brk_path+"yyy_a_long_trial_36.png",bbox_inches='tight')













dual_alpha = 0.4
hatch_alpha = 1.0
hatch_string = '||'
hatch_strings = ['\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\', '\\\\']
#hatch_strings = ['\\\\', 'o', '//', '++', '..']
hatch_strings = ['\\\\', '.', '\\\\', '||', '\\\\', '.', '\\\\', '||']
solo_alpha = 0.4
lw_for_plot = 1.0

hatch_a = '|||'
hatch_b = '|||'

fill_color = '#f1f1f1'
new_grey = '#b46e0f'

#hack: FIG: yyy_b_long_trial_36
print "yyy_b_long_trial_36"
fig=plt.figure(figsize=(32.0,7.5))
rainbow_lw = 1.0



nub = range(len(temp_string_list))


for iii in nub[2:]:

    ax=fig.add_subplot(2, len(temp_string_list), iii+1, frameon=True)


    # ax.fill_between(distance_vector, compound_alt_vol_dual_max[:,0,iii], compound_alt_vol_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_a, edgecolor=plot_col_hatch[iii+1])
    #
    # plt.plot(distance_vector, compound_alt_vol_solo[:,:,iii], lw=0.5, color=plot_col[iii+1])
    # #plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])


    jump = ax.fill_between(distance_vector, compound_alt_vol_b_max_shift[:,0,iii], compound_alt_vol_b_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_b, edgecolor=new_grey)

    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,:,iii], lw=0.5, color=new_grey)


    plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
    for j in range(nsites):
        plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

    plt.xlim([20000.0,110000.0])
    plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
    plt.ylim([-1.0,30.0])
    plt.title(temp_string_list[iii])
    # plt.xlabel("crust age [Myr]", fontsize=9)
    # plt.ylabel("alteration volume percent", fontsize=9)
    #plt.legend(fontsize=8,loc='best',ncol=1)








hatch_strings = ['//', '//', '//', '//', '//','//', '//', '//', '//', '//']


ax=fig.add_subplot(2, len(temp_string_list), len(temp_string_list) + iii+1, frameon=True)


for iii in nub[2:]:


    ax=fig.add_subplot(2, len(temp_string_list), len(temp_string_list) + iii+1, frameon=True)

    # ax.fill_between(distance_vector, compound_alt_fe_dual_max[:,0,iii], compound_alt_fe_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_a, edgecolor=plot_col_hatch[iii+1])
    #
    # plt.plot(distance_vector, compound_alt_fe_solo[:,:,iii], lw=0.5, color=plot_col[iii+1])


    ax.fill_between(distance_vector, compound_alt_fe_b_max_shift[:,0,iii], compound_alt_fe_b_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_b, edgecolor=new_grey)

    plt.plot(distance_vector, compound_alt_fe_solo_shift[:,:,iii], lw=0.5, color=new_grey)


    plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
    for j in range(nsites):
        plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
    ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

    plt.xlim([20000.0,110000.0])
    plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
    plt.ylim([0.6,0.8])
    # plt.ylabel("FeO/FeOt", fontsize=9)
    #plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))




plt.savefig(brk_path+"yyy_b_long_trial_36.png",bbox_inches='tight')














#hack: FIG: yyy_ab_long_trial_36
print "yyy_ab_long_trial_36"
fig=plt.figure(figsize=(48,11.25))
# fig=plt.figure(figsize=(32.0,7.5))
rainbow_lw = 1.0



nub = range(len(temp_string_list))

#block_scale = 0.856

for iii in nub[2:-1]:

    ax=fig.add_subplot(2, len(temp_string_list), iii+1, frameon=True)

    intercept_toggle = 0.0


    # ax.fill_between(distance_vector, compound_alt_vol_dual_max[:,0,iii], compound_alt_vol_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_a, edgecolor=plot_col_hatch[iii+1])
    #
    # plt.plot(distance_vector, compound_alt_vol_solo[:,:,iii], lw=0.5, color=plot_col[iii+1])
    # #plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])

    new_grey = '#317783'



    jump = ax.fill_between(distance_vector, compound_alt_vol_a_max_shift[:,0,iii]+the_intercept*intercept_toggle, compound_alt_vol_a_min_shift[:,0,iii]+the_intercept*intercept_toggle, facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_b, edgecolor=new_grey)

    # print " "
    # print "AB stats"
    # print "distance_vector" , distance_vector
    # print " "
    # print "compound_alt_vol_a_max_shift[:,0,iii]"
    # print compound_alt_vol_a_max_shift[:,0,iii]
    # print " "

    new_grey = '#b46e0f'

    jump = ax.fill_between(distance_vector, compound_alt_vol_b_max_shift[:,0,iii]+the_intercept*intercept_toggle, compound_alt_vol_b_min_shift[:,0,iii]+the_intercept*intercept_toggle, facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_b, edgecolor=new_grey)

    # print " "
    # print "compound_alt_vol_b_max_shift[:,0,iii]"
    # print compound_alt_vol_b_max_shift[:,0,iii]
    # print " "

    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,:,iii]+the_intercept*intercept_toggle, lw=0.5, color=new_grey)


    plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
    for j in range(nsites):
        plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

    print "attempt at scaling the_slope"
    print the_slope/(3.677e-5)
    print " "
    print "lower" , (the_slope-(2.64e-5)-(2.64e-5))/(3.677e-5)
    print "upper" , (the_slope+(2.64e-5)+(2.64e-5))/(3.677e-5)



    plt.plot(site_locations, site_locations*the_slope + the_intercept*intercept_toggle, lw=2.0, color='k')
    plt.plot(site_locations,site_locations*(the_slope+(2.64e-5)+(2.64e-5)) + the_intercept*intercept_toggle, lw=1.0, color='k')
    plt.plot(site_locations,site_locations*(the_slope-(2.64e-5)-(2.64e-5)) + the_intercept*intercept_toggle, lw=1.0, color='k')

    plt.plot(site_locations, site_locations*1.0*(3.677e-5), lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*2.0*(3.677e-5), lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*3.0*(3.677e-5), lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*4.0*(3.677e-5), lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*5.0*(3.677e-5), lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*6.0*(3.677e-5), lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*7.0*(3.677e-5), lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*8.0*(3.677e-5), lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*9.0*(3.677e-5), lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*10.0*(3.677e-5), lw=1.0, color='m',zorder=55)

    plt.plot(site_locations, site_locations*4.71*(3.677e-5), lw=1.5, color='g',zorder=55)
    plt.plot(site_locations, site_locations*9.6*(3.677e-5), lw=1.5, color='g',zorder=55)

    plt.plot(site_locations, site_locations*0.14*(3.677e-5), lw=1.5, linestyle='--', color='g',zorder=55)
    plt.plot(site_locations, site_locations*11.25*(3.677e-5), lw=1.5, linestyle='--', color='g',zorder=55)

    plt.plot(site_locations, site_locations*0.127*(3.677e-5), lw=1.5, linestyle=':', color='b',zorder=55)
    plt.plot(site_locations, site_locations*9.628*(3.677e-5), lw=1.5, linestyle=':', color='b',zorder=55)

    plt.xlim([20000.0,110000.0])
    plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
    plt.ylim([-1.0,30.0])
    plt.title(temp_string_list[iii])
    # plt.xlabel("crust age [Myr]", fontsize=9)
    # plt.ylabel("alteration volume percent", fontsize=9)
    #plt.legend(fontsize=8,loc='best',ncol=1)








hatch_strings = ['//', '//', '//', '//', '//','//', '//', '//', '//', '//']


ax=fig.add_subplot(2, len(temp_string_list), len(temp_string_list) + iii+1, frameon=True)


for iii in nub[2:-1]:


    ax=fig.add_subplot(2, len(temp_string_list), len(temp_string_list) + iii+1, frameon=True)

    # ax.fill_between(distance_vector, compound_alt_fe_dual_max[:,0,iii], compound_alt_fe_dual_min[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_a, edgecolor=plot_col_hatch[iii+1])
    #
    # plt.plot(distance_vector, compound_alt_fe_solo[:,:,iii], lw=0.5, color=plot_col[iii+1])


    new_grey = '#317783'

    ax.fill_between(distance_vector, compound_alt_fe_a_max_shift[:,0,iii], compound_alt_fe_a_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_b, edgecolor=new_grey)

    # print " "
    # print "AB stats"
    # print "distance_vector" , distance_vector
    # print " "
    # print "compound_alt_fe_a_max_shift[:,0,iii]"
    # print compound_alt_fe_a_max_shift[:,0,iii]
    # print " "

    new_grey = '#b46e0f'

    ax.fill_between(distance_vector, compound_alt_fe_b_max_shift[:,0,iii], compound_alt_fe_b_min_shift[:,0,iii], facecolor='none', lw=0, zorder=15-iii, alpha=hatch_alpha, hatch=hatch_b, edgecolor=new_grey)

    # print " "
    # print "compound_alt_fe_b_max_shift[:,0,iii]"
    # print compound_alt_fe_b_max_shift[:,0,iii]
    # print " "

    plt.plot(distance_vector, compound_alt_fe_solo_shift[:,:,iii], lw=0.5, color=new_grey)



    plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
    for j in range(nsites):
        plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
        plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
    ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)


    intercept_toggle_fe = 0.0

    # plt.plot(site_locations, site_locations*the_slope_fe + (the_intercept_fe - (intercept_toggle_fe*0.78)), lw=2.0, color='k')
    # plt.plot(site_locations,site_locations*(the_slope_fe+(3.32e-7)+(3.32e-7)) + (the_intercept_fe - (intercept_toggle_fe*0.78)), lw=1.0, color='k')
    # plt.plot(site_locations,site_locations*(the_slope_fe-(3.32e-7)-(3.32e-7)) + (the_intercept_fe - (intercept_toggle_fe*0.78)), lw=1.0, color='k')

    plt.plot(site_locations, site_locations*the_slope_fe + 0.78, lw=2.0, color='k')
    plt.plot(site_locations,site_locations*(the_slope_fe+(3.32e-7)+(3.32e-7)) + 0.78, lw=1.0, color='k')
    plt.plot(site_locations,site_locations*(the_slope_fe-(3.32e-7)-(3.32e-7)) + 0.78, lw=1.0, color='k')

    print "attempt at scaling the_slope"
    print the_slope_fe/(3.677e-5)
    print " "
    print "lower" , (the_slope_fe-(3.32e-7)-(3.32e-7))/(3.677e-5)
    print "upper" , (the_slope_fe+(3.32e-7)+(3.32e-7))/(3.677e-5)


    plt.plot(site_locations, site_locations*(-0.01)*(3.677e-5) + 0.78, lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*(-0.02)*(3.677e-5) + 0.78, lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*(-0.03)*(3.677e-5) + 0.78, lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*(-0.04)*(3.677e-5) + 0.78, lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*(-0.05)*(3.677e-5) + 0.78, lw=1.0, color='m',zorder=55)
    plt.plot(site_locations, site_locations*(-0.06)*(3.677e-5) + 0.78, lw=1.0, color='m',zorder=55)


    plt.plot(site_locations, site_locations*(-0.00748)*(3.677e-5) + 0.78, lw=1.5, color='g',zorder=55)
    plt.plot(site_locations, site_locations*(-0.0347)*(3.677e-5) + 0.78, lw=1.5, color='g',zorder=55)

    plt.plot(site_locations, site_locations*(-0.0008819)*(3.677e-5) + 0.78, lw=1.5, linestyle='--', color='g',zorder=55)
    plt.plot(site_locations, site_locations*(-0.05539)*(3.677e-5) + 0.78, lw=1.5, linestyle='--', color='g',zorder=55)

    plt.plot(site_locations, site_locations*(-0.00075)*(3.677e-5) + 0.78, lw=1.5, linestyle=':', color='b',zorder=55)
    plt.plot(site_locations, site_locations*(-0.04741)*(3.677e-5) + 0.78, lw=1.5, linestyle=':', color='b',zorder=55)

    plt.xlim([20000.0,110000.0])
    plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
    plt.ylim([0.6,0.8])
    # plt.ylabel("FeO/FeOt", fontsize=9)
    #plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))




plt.savefig(brk_path+"yyy_ab_long_trial_36.png",bbox_inches='tight')


##hack: temp shift!
compound_alt_vol_solo = compound_alt_vol_solo_shift
compound_alt_fe_solo = compound_alt_fe_solo_shift







sp1 = 4
sp2 = len(temp_string_list)
cont_cmap = cm.rainbow
xskip = 4
yskip = 2
bar_bins = 4

cont_x_diff_max = len(diff_strings) - 0
cont_y_param_max = len(param_strings) - 0




#poop: dimensions
dim1 = 12.0
dim2 = 14.0











#poop: FIG: ai_CHUNKS_36
print "ai_CHUNKS_36"
# new ratio is d_pri : alt-fe

sp11 = 4
sp22 = len(temp_string_list)

fig=plt.figure(figsize=(12.0,12.0))
#plt.subplots_adjust(hspace=0.4)

cmap_divide = cont_y_param_max

blue_cmap = LinearSegmentedColormap.from_list("blue_colormap", ((0.1, 0.8, 1.0), (0.0, 0.0, 0.5)), N=30, gamma=1.0)
blue_colors = [ blue_cmap(x) for x in np.linspace(0.0, 1.0, cmap_divide) ]

red_cmap = LinearSegmentedColormap.from_list("red_colormap", ((0.8, 0.5, 0.0), (0.5, 0.0, 0.0)), N=30, gamma=1.0)
red_colors = [ red_cmap(x) for x in np.linspace(0.0, 1.0, cmap_divide) ]

for iii in range(len(temp_string_list)):

    # the_s = np.abs(value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    # the_d = np.abs(value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    # the_a = np.abs(value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    # the_b = np.abs(value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])
    # the_b = the_a
    #
    the_s_row = np.abs(value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,:]) #/value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,:]
    #the_b_row = the_a_row

    min_all_alt_fe = np.min(the_s_row)
    if np.min(the_d_row) < min_all_alt_fe:
        min_all_alt_fe = np.min(the_d_row)
    if np.min(the_a_row) < min_all_alt_fe:
        min_all_alt_fe = np.min(the_a_row)
    if np.min(the_b_row) < min_all_alt_fe:
        min_all_alt_fe = np.min(the_b_row)

    max_all_alt_fe = np.max(the_s_row)
    if np.max(the_d_row) > max_all_alt_fe:
        max_all_alt_fe = np.max(the_d_row)
    if np.max(the_a_row) > max_all_alt_fe:
        max_all_alt_fe = np.max(the_a_row)
    if np.max(the_b_row) > max_all_alt_fe:
        max_all_alt_fe = np.max(the_b_row)


    the_s_row = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,:]) #/value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,:]
    #the_b_row = the_a_row

    min_all_alt_fe = np.min(the_s_row)
    if np.min(the_d_row) < min_all_alt_fe:
        min_all_alt_fe = np.min(the_d_row)
    if np.min(the_a_row) < min_all_alt_fe:
        min_all_alt_fe = np.min(the_a_row)
    if np.min(the_b_row) < min_all_alt_fe:
        min_all_alt_fe = np.min(the_b_row)

    max_all_dpri_mean = np.max(the_s_row)
    if np.max(the_d_row) > max_all_dpri_mean:
        max_all_dpri_mean = np.max(the_d_row)
    if np.max(the_a_row) > max_all_dpri_mean:
        max_all_dpri_mean = np.max(the_a_row)
    if np.max(the_b_row) > max_all_dpri_mean:
        max_all_dpri_mean = np.max(the_b_row)


    min_all = -0.05
    max_all = 1.05

    q_skip = 2

    ax=fig.add_subplot(sp22, sp11, iii*sp11 + 1, frameon=True)
    for ii in range(len(param_strings)):

        # red = alt_fe_mean
        plt.plot(diff_nums,value_alt_fe_mean[ii,:,iii]/max_all_alt_fe,color=red_colors[ii],zorder=3)
        # blue = dpri_mean
        plt.plot(diff_nums,np.abs(value_dpri_mean[ii,:,iii])/max_all_dpri_mean,color=blue_colors[ii],zorder=5)

    plt.title('temp = ' + temp_string_list[iii] + 'chamber s')
    plt.ylim([min_all,max_all])
    if iii != len(temp_string_list)-1:
        plt.xticks([])



    ax=fig.add_subplot(sp22, sp11, iii*sp11 + 2, frameon=True)
    for ii in range(len(param_strings)):
        # red = alt_fe_mean_d
        # plt.plot(diff_nums,value_alt_fe_mean_d[ii,:,iii]/np.max(value_alt_fe_mean_d[ii,:,iii]),color='r',zorder=3)
        plt.plot(diff_nums,value_alt_fe_mean_d[ii,:,iii]/max_all_alt_fe,color=red_colors[ii],zorder=3)
        # blue = dpri
        plt.plot(diff_nums,np.abs(value_dpri_mean_d[ii,:,iii])/max_all_dpri_mean,color=blue_colors[ii],zorder=5)
    plt.title('temp = ' + temp_string_list[iii] + 'chamber d')
    plt.ylim([min_all,max_all])
    if iii != len(temp_string_list)-1:
        plt.xticks([])

    ax=fig.add_subplot(sp22, sp11, iii*sp11 + 3, frameon=True)
    for ii in range(len(param_strings)):
        # red = alt_fe_mean_d
        # plt.plot(diff_nums,value_alt_fe_mean_a[ii,:,iii]/np.max(value_alt_fe_mean_a[ii,:,iii]),color='r',zorder=3)
        plt.plot(diff_nums,value_alt_fe_mean_a[ii,:,iii]/max_all_alt_fe,color=red_colors[ii],zorder=3)
        # blue = dpri
        plt.plot(diff_nums,np.abs(value_dpri_mean_a[ii,:,iii])/max_all_dpri_mean,color=blue_colors[ii],zorder=5)
    plt.title('temp = ' + temp_string_list[iii] + 'chamber a')
    plt.ylim([min_all,max_all])
    if iii != len(temp_string_list)-1:
        plt.xticks([])

    ax=fig.add_subplot(sp22, sp11, iii*sp11 + 4, frameon=True)
    for ii in range(len(param_strings)):
        # red = alt_fe_mean_d
        # plt.plot(diff_nums,value_alt_fe_mean_b[ii,:,iii]/np.max(value_alt_fe_mean_b[ii,:,iii]),color='r',zorder=3)
        plt.plot(diff_nums,value_alt_fe_mean_b[ii,:,iii]/max_all_alt_fe,color=red_colors[ii],zorder=3)
        # blue = dpri
        plt.plot(diff_nums,np.abs(value_dpri_mean_b[ii,:,iii])/max_all_dpri_mean,color=blue_colors[ii],zorder=5)
    plt.title('temp = ' + temp_string_list[iii] + 'chamber b')
    plt.ylim([min_all,max_all])
    if iii != len(temp_string_list)-1:
        plt.xticks([])



plt.savefig(brk_path+"z_ai_CHUNKS_36.png",bbox_inches='tight')










#hack: FIG: ai_calc_to_dpri
print "ai_calc_to_dpri"
save_string = "calc_to_dpri "


fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    the_s = np.abs(value_dsec[:cont_y_param_max,:cont_x_diff_max,9,iii]/value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(value_dsec_d[:cont_y_param_max,:cont_x_diff_max,9,iii]/value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(value_dsec_a[:cont_y_param_max,:cont_x_diff_max,9,iii]/value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(value_dsec_b[:cont_y_param_max,:cont_x_diff_max,9,iii]/value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = the_a

    the_s_row = np.abs(value_dsec[:cont_y_param_max,:cont_x_diff_max,9,:]/value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(value_dsec_d[:cont_y_param_max,:cont_x_diff_max,9,:]/value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(value_dsec_a[:cont_y_param_max,:cont_x_diff_max,9,:]/value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(value_dsec_b[:cont_y_param_max,:cont_x_diff_max,9,:]/value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = the_a_row

    # the_s_row = the_s
    # the_d_row = the_d
    # the_a_row = the_a
    # the_b_row = the_b

    min_all = np.min(the_s_row)
    if np.min(the_d_row) < min_all:
        min_all = np.min(the_d_row)
    if np.min(the_a_row) < min_all:
        min_all = np.min(the_a_row)
    if np.min(the_b_row) < min_all:
        min_all = np.min(the_b_row)

    max_all = np.max(the_s_row)
    if np.max(the_d_row) > max_all:
        max_all = np.max(the_d_row)
    if np.max(the_a_row) > max_all:
        max_all = np.max(the_a_row)
    if np.max(the_b_row) > max_all:
        max_all = np.max(the_b_row)

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s " + save_string + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(brk_path+"z_ai_calc_to_dpri.png",bbox_inches='tight')







#poop: THE 2D SAVE JUNE FIGS!
tempy_index_shift = 2


#hack: FIG: only_feot
print "only_feot"

fig=plt.figure(figsize=(9.0,14.0))
plt.subplots_adjust(hspace=0.2)

for iii in range(len(temp_string_list[tempy_index_shift:-1])):

    iii = iii + tempy_index_shift

    the_s = np.abs(save_feot_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = 0.5*np.abs(save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(save_feot_mean[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_d_row = 0.5*np.abs(save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_a_row = np.abs(save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_b_row = np.abs(save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])

    min_all = np.min(the_s_row)
    if np.min(the_d_row) < min_all:
        min_all = np.min(the_d_row)
    if np.min(the_a_row) < min_all:
        min_all = np.min(the_a_row)
    if np.min(the_b_row) < min_all:
        min_all = np.min(the_b_row)

    max_all = np.max(the_s_row)
    if np.max(the_d_row) > max_all:
        max_all = np.max(the_d_row)
    if np.max(the_a_row) > max_all:
        max_all = np.max(the_a_row)
    if np.max(the_b_row) > max_all:
        max_all = np.max(the_b_row)

    # min_all = 1.0
    # max_all = 17.0

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s only_feot " + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d only_feot " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a only_feot " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b only_feot " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(brk_path+"only_feot.png",bbox_inches='tight')




#hack: FIG: only_dpri_feot
print "only_dpri_feot"

fig=plt.figure(figsize=(9.0,14.0))
plt.subplots_adjust(hspace=0.2)

for iii in range(len(temp_string_list[tempy_index_shift:-1])):

    iii = iii + tempy_index_shift

    the_s = np.abs(save_feot_mean[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = 0.5*np.abs(save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(save_feot_mean[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_d_row = 0.5*np.abs(save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_a_row = np.abs(save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_b_row = np.abs(save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])

    the_b = the_a
    the_b_row = the_a_row

    min_all = np.min(the_s_row)
    if np.min(the_d_row) < min_all:
        min_all = np.min(the_d_row)
    if np.min(the_a_row) < min_all:
        min_all = np.min(the_a_row)
    if np.min(the_b_row) < min_all:
        min_all = np.min(the_b_row)

    max_all = np.max(the_s_row)
    if np.max(the_d_row) > max_all:
        max_all = np.max(the_d_row)
    if np.max(the_a_row) > max_all:
        max_all = np.max(the_a_row)
    if np.max(the_b_row) > max_all:
        max_all = np.max(the_b_row)

    # min_all = 1.0
    # max_all = 17.0

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s only_dpri_feot " + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d only_dpri_feot " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a only_dpri_feot " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b only_dpri_feot " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(brk_path+"only_dpri_feot.png",bbox_inches='tight')







#hack: FIG: only_mgo
print "only_mgo"
save_string = "only_mgo "

fig=plt.figure(figsize=(9.0,14.0))
plt.subplots_adjust(hspace=0.2)

for iii in range(len(temp_string_list[tempy_index_shift:-1])):

    iii = iii + tempy_index_shift

    the_s = np.abs(save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = 0.5*np.abs(save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_d_row = 0.5*np.abs(save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_a_row = np.abs(save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_b_row = np.abs(save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])

    min_all = np.min(the_s_row)
    if np.min(the_d_row) < min_all:
        min_all = np.min(the_d_row)
    if np.min(the_a_row) < min_all:
        min_all = np.min(the_a_row)
    if np.min(the_b_row) < min_all:
        min_all = np.min(the_b_row)

    max_all = np.max(the_s_row)
    if np.max(the_d_row) > max_all:
        max_all = np.max(the_d_row)
    if np.max(the_a_row) > max_all:
        max_all = np.max(the_a_row)
    if np.max(the_b_row) > max_all:
        max_all = np.max(the_b_row)

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s " + save_string + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(brk_path+"only_mgo.png",bbox_inches='tight')





#hack: FIG: only_dpri_mgo
print "only_dpri_mgo"
save_string = "only_dpri_mgo "

fig=plt.figure(figsize=(9.0,14.0))
plt.subplots_adjust(hspace=0.2)

for iii in range(len(temp_string_list[tempy_index_shift:-1])):

    iii = iii + tempy_index_shift

    the_s = np.abs(save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = 0.5*np.abs(save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_d_row = 0.5*np.abs(save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_a_row = np.abs(save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_b_row = np.abs(save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])

    the_b = the_a
    the_b_row = the_a_row

    min_all = np.min(the_s_row)
    if np.min(the_d_row) < min_all:
        min_all = np.min(the_d_row)
    if np.min(the_a_row) < min_all:
        min_all = np.min(the_a_row)
    if np.min(the_b_row) < min_all:
        min_all = np.min(the_b_row)

    max_all = np.max(the_s_row)
    if np.max(the_d_row) > max_all:
        max_all = np.max(the_d_row)
    if np.max(the_a_row) > max_all:
        max_all = np.max(the_a_row)
    if np.max(the_b_row) > max_all:
        max_all = np.max(the_b_row)

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s " + save_string + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(brk_path+"only_dpri_mgo.png",bbox_inches='tight')






#hack: FIG: only_dpri_k2o
print "only_k2o"
save_string = "only_k2o "

fig=plt.figure(figsize=(9.0,14.0))
plt.subplots_adjust(hspace=0.2)

for iii in range(len(temp_string_list[tempy_index_shift:-1])):

    iii = iii + tempy_index_shift

    the_s = np.abs(save_k2o_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = 0.5*np.abs(save_k2o_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(save_k2o_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(save_k2o_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(save_k2o_mean[:cont_y_param_max,:cont_x_diff_max,:-1])
    the_d_row = 0.5*np.abs(save_k2o_mean_d[:cont_y_param_max,:cont_x_diff_max,:-1])
    the_a_row = np.abs(save_k2o_mean_a[:cont_y_param_max,:cont_x_diff_max,:-1])
    the_b_row = np.abs(save_k2o_mean_b[:cont_y_param_max,:cont_x_diff_max,:-1])

    min_all = np.min(the_s_row)
    if np.min(the_d_row) < min_all:
        min_all = np.min(the_d_row)
    if np.min(the_a_row) < min_all:
        min_all = np.min(the_a_row)
    if np.min(the_b_row) < min_all:
        min_all = np.min(the_b_row)

    max_all = np.max(the_s_row)
    if np.max(the_d_row) > max_all:
        max_all = np.max(the_d_row)
    if np.max(the_a_row) > max_all:
        max_all = np.max(the_a_row)
    if np.max(the_b_row) > max_all:
        max_all = np.max(the_b_row)

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s " + save_string + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(brk_path+"only_k2o.png",bbox_inches='tight')

#374046




#hack: FIG: only_k2o
print "only_dpri_k2o"
save_string = "only_dpri_k2o "

fig=plt.figure(figsize=(9.0,14.0))
plt.subplots_adjust(hspace=0.2)

for iii in range(len(temp_string_list[tempy_index_shift:-1])):

    iii = iii + tempy_index_shift

    the_s = np.abs(save_k2o_mean[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = 0.5*np.abs(save_k2o_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(save_k2o_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(save_k2o_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])/np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(save_k2o_mean[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_d_row = 0.5*np.abs(save_k2o_mean_d[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_a_row = np.abs(save_k2o_mean_a[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])
    the_b_row = np.abs(save_k2o_mean_b[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])/np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,tempy_index_shift:-1])

    min_all = np.min(the_s_row)
    if np.min(the_d_row) < min_all:
        min_all = np.min(the_d_row)
    if np.min(the_a_row) < min_all:
        min_all = np.min(the_a_row)
    if np.min(the_b_row) < min_all:
        min_all = np.min(the_b_row)

    max_all = np.max(the_s_row)
    if np.max(the_d_row) > max_all:
        max_all = np.max(the_d_row)
    if np.max(the_a_row) > max_all:
        max_all = np.max(the_a_row)
    if np.max(the_b_row) > max_all:
        max_all = np.max(the_b_row)

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s " + save_string + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(brk_path+"only_dpri_k2o.png",bbox_inches='tight')
