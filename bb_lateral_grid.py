# bb_lateral_grid.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 11
from matplotlib.colors import LinearSegmentedColormap

plot_col = ['#000000', '#940000', '#d26618', '#dfa524', '#9ac116', '#139a31', '#35b5aa', '#0740d2', '#7f05d4', '#b100de', '#fba8ff']

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






def square_contour(sp1, sp2, sp, cont_block, cb_title="", xlab=0, ylab=0, cont_levels_in=[1.0,2.0]):
    ax2=fig.add_subplot(sp1, sp2, sp, frameon=True)

    if xlab == 1:
        plt.xlabel('log10(mixing time [years])', fontsize=8)
    if ylab == 1:
        plt.ylabel('discharge q [m/yr]', fontsize=8)

    pCont = ax2.contourf(x_grid,y_grid,cont_block, levels=cont_levels_in, cmap=cont_cmap, antialiased=True, linewidth=0.0)
    for c in pCont.collections:
        c.set_edgecolor("face")

    # plt.contour(x_grid, y_grid, cont_block, levels=[0.0, 4.7, 5.38, 10.2], colors='w', linewidth=6.0)

    plt.xticks(diff_nums[:cont_x_diff_max:xskip],diff_strings[::xskip], fontsize=8)
    plt.yticks(param_nums[:cont_y_param_max:yskip],param_strings[::yskip], fontsize=8)

    plt.title(cb_title, fontsize=9)

    bbox = ax2.get_position()
    cax = fig.add_axes([bbox.xmin+0.0, bbox.ymin-0.05, bbox.width*1.0, bbox.height*0.05])
    cbar = plt.colorbar(pCont, cax = cax,orientation='horizontal')
    cbar.set_ticks(cont_levels_in[::cont_skip])
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_xlabel(cb_title,fontsize=9,labelpad=clabelpad)
    cbar.solids.set_edgecolor("face")

    return square_contour



def square_contour_min(sp1, sp2, sp, cont_block, cb_title="", xlab=0, ylab=0, the_cbar=0, cont_levels_in=[1.0,2.0]):
    ax2=fig.add_subplot(sp1, sp2, sp, frameon=True)

    if xlab == 1:
        plt.xlabel('log10(mixing time [years])', fontsize=8)
    if ylab == 1:
        plt.ylabel('discharge q [m/yr]', fontsize=8)

    pCont = ax2.contourf(x_grid,y_grid,cont_block, levels=cont_levels_in, cmap=cont_cmap, antialiased=True, linewidth=0.0)
    for c in pCont.collections:
        c.set_edgecolor("face")

    plt.xticks(diff_nums[:cont_x_diff_max:xskip],diff_strings[::xskip], fontsize=8)
    plt.yticks(param_nums[:cont_y_param_max:yskip],param_strings[::yskip], fontsize=8)

    plt.title(cb_title, fontsize=9)

    if the_cbar == 1:
        bbox = ax2.get_position()
        cax = fig.add_axes([bbox.xmin+0.25, bbox.ymin-0.05, bbox.width*2.0, bbox.height*0.04])
        cbar = plt.colorbar(pCont, cax = cax,orientation='horizontal')
        cbar.set_ticks(cont_levels_in[::cont_skip])
        cbar.ax.tick_params(labelsize=7)
        # cbar.ax.set_xlabel(cb_title,fontsize=9,labelpad=clabelpad)
        cbar.solids.set_edgecolor("face")

    return square_contour_min



def any_2d_interp(x_in, y_in, z_in, x_diff_path, y_param_path, kind_in='linear'):

    the_f = interpolate.interp2d(x_in, y_in, z_in, kind=kind_in)
    any_2d_interp = the_f(x_diff_path, y_param_path)

    return any_2d_interp


#todo: path + params
temp_string = "30"
in_path = "../output/revival/winter_basalt_box/"
dir_path = "z_group_dd_full_"+temp_string+"/"
fig_path = "fig_lateral/"

param_strings = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']
param_nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# param_strings = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0']
# param_nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

diff_strings = ['2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50', '3.75', '4.00', '4.25', '4.50']
diff_nums = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5]


#poop: make 2d alt_ind grids
n_grids = 3

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


#poop: make value_sec_x grids
minNum = 41
value_dsec = np.zeros([len(param_strings),len(diff_strings),minNum+1,n_grids])
value_dsec_d = np.zeros([len(param_strings),len(diff_strings),minNum+1,n_grids])
value_dsec_a = np.zeros([len(param_strings),len(diff_strings),minNum+1,n_grids])
value_dsec_b = np.zeros([len(param_strings),len(diff_strings),minNum+1,n_grids])

#009BC7
#c7c7c7
#80BBFF
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


#todo: LOAD IN 2d alt index grids
value_alt_vol_mean[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean.txt')
value_alt_vol_mean_d[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean_d.txt')
value_alt_vol_mean_a[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean_a.txt')
value_alt_vol_mean_b[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean_b.txt')

value_alt_fe_mean[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean.txt')
value_alt_fe_mean_d[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean_d.txt')
value_alt_fe_mean_a[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean_a.txt')
value_alt_fe_mean_b[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean_b.txt')

value_dpri_mean[:,:,0] = np.loadtxt(in_path + dir_path + 'value_dpri_mean.txt')
value_dpri_mean_d[:,:,0] = np.loadtxt(in_path + dir_path + 'value_dpri_mean_d.txt')
value_dpri_mean_a[:,:,0] = np.loadtxt(in_path + dir_path + 'value_dpri_mean_a.txt')
value_dpri_mean_b[:,:,0] = np.loadtxt(in_path + dir_path + 'value_dpri_mean_b.txt')


#todo: LOAD IN value_sec_x.txt

secondary = np.array(['', 'kaolinite', 'saponite_mg', 'celadonite', 'clinoptilolite', 'pyrite', 'mont_na', 'goethite',
'smectite', 'calcite', 'kspar', 'saponite_na', 'nont_na', 'nont_mg', 'fe_celad', 'nont_ca',
'mesolite', 'hematite', 'mont_ca', 'verm_ca', 'analcime', 'philipsite', 'mont_mg', 'gismondine',
'verm_mg', 'natrolite', 'talc', 'smectite_low', 'prehnite', 'chlorite', 'scolecite', 'clinochlorte14a',
'clinochlore7a', 'saponite_ca', 'verm_na', 'pyrrhotite', 'fe_saponite_ca', 'fe_saponite_mg', 'daphnite7a', 'daphnite14a', 'epidote'])


any_min = []
for j in range(1,minNum):
    if os.path.isfile(in_path + dir_path + 'value_dsec_'+str(int(j))+'.txt'):
        if not np.any(any_min == j):
            any_min = np.append(any_min,j)
        value_dsec[:,:,j,0] = np.loadtxt(in_path + dir_path + 'value_dsec_'+str(int(j))+'.txt')

    if os.path.isfile(in_path + dir_path + 'value_dsec_'+str(int(j))+'_d.txt'):
        if not np.any(any_min == j):
            any_min = np.append(any_min,j)
        value_dsec_d[:,:,j,0] = np.loadtxt(in_path + dir_path + 'value_dsec_'+str(int(j))+'_d.txt')

    if os.path.isfile(in_path + dir_path + 'value_dsec_'+str(int(j))+'_a.txt'):
        if not np.any(any_min == j):
            any_min = np.append(any_min,j)
        value_dsec_a[:,:,j,0] = np.loadtxt(in_path + dir_path + 'value_dsec_'+str(int(j))+'_a.txt')

    if os.path.isfile(in_path + dir_path + 'value_dsec_'+str(int(j))+'_b.txt'):
        if not np.any(any_min == j):
            any_min = np.append(any_min,j)
        value_dsec_b[:,:,j,0] = np.loadtxt(in_path + dir_path + 'value_dsec_'+str(int(j))+'_b.txt')

print "any_min: " , any_min

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

site_locations = np.array([22.742, 25.883, 33.872, 40.706, 45.633, 55.765, 75.368, 99.006, 102.491])
site_locations = (site_locations - 00.00)*1000.0

site_names = ["1023", "1024", "1025", "1031", "1028", "1029", "1032", "1026", "1027"]
alt_values = np.array([0.3219, 2.1072, 2.3626, 2.9470, 10.0476, 4.2820, 8.9219, 11.8331, 13.2392])
lower_eb = np.array([0.3219, 0.04506, 0.8783, 1.7094, 5.0974, 0.8994, 5.3745, 2.5097, 3.0084])
upper_eb = np.array([1.7081, 2.9330, 3.7662, 4.9273, 11.5331, 5.0247, 10.7375, 17.8566, 27.4308])

fe_values = np.array([0.7753, 0.7442, 0.7519, 0.7610, 0.6714, 0.7416, 0.7039, 0.6708, 0.6403])
lower_eb_fe = np.array([0.7753, 0.7442, 0.7208, 0.7409, 0.6240, 0.7260, 0.6584, 0.6299, 0.6084])
upper_eb_fe = np.array([0.7753, 0.7442, 0.7519, 0.7812, 0.7110, 0.7610, 0.7396, 0.7104, 0.7026])




#hack: lateral grid
n_lateral = 100
n_compounds = 10
age_vector = np.linspace(0.8,3.5,n_lateral,endpoint=True)
distance_vector = np.linspace(00000.0, 120000.0,n_lateral, endpoint=True)
compound_alt_vol = np.zeros([n_lateral,n_compounds])
compound_alt_fe = np.zeros([n_lateral,n_compounds])

compound_alt_vol_shift = np.zeros([n_lateral,n_compounds])
compound_alt_fe_shift = np.zeros([n_lateral,n_compounds])

compound_alt_vol_labels = [3.0, 5.0, 10.0, 15.0, 20.0]
compound_alt_fe_labels = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08]

compound_alt_vol_solo = np.zeros([n_lateral, len(param_strings)])
compound_alt_fe_solo = np.zeros([n_lateral, len(param_strings)])

compound_alt_vol_solo_shift = np.zeros([n_lateral, len(param_strings)])
compound_alt_fe_solo_shift = np.zeros([n_lateral, len(param_strings)])

for i in range(n_lateral-1):
    # compound at 3.0 % / Myr
    compound_alt_vol[i,0] = (2.7/n_lateral)*i*compound_alt_vol_labels[0]
    # compound at 5.0 % / Myr
    compound_alt_vol[i,1] = (2.7/n_lateral)*i*compound_alt_vol_labels[1]
    # compound at 10.0 % / Myr
    compound_alt_vol[i,2] = (2.7/n_lateral)*i*compound_alt_vol_labels[2]
    # compound at 15.0 % / Myr
    compound_alt_vol[i,3] = (2.7/n_lateral)*i*compound_alt_vol_labels[3]
    # compound at 20.0 % / Myr
    compound_alt_vol[i,4] = (2.7/n_lateral)*i*compound_alt_vol_labels[4]

    # compound at 10.0 % / Myr
    if age_vector[i] > 0.5:
        compound_alt_vol_shift[i,0] = (2.7/n_lateral)*i*compound_alt_vol_labels[0] - 0.5*compound_alt_vol_labels[0]
        compound_alt_vol_shift[i,1] = (2.7/n_lateral)*i*compound_alt_vol_labels[1] - 0.5*compound_alt_vol_labels[1]
        compound_alt_vol_shift[i,2] = (2.7/n_lateral)*i*compound_alt_vol_labels[2] - 0.5*compound_alt_vol_labels[2]
        compound_alt_vol_shift[i,3] = (2.7/n_lateral)*i*compound_alt_vol_labels[3] - 0.5*compound_alt_vol_labels[3]
        compound_alt_vol_shift[i,4] = (2.7/n_lateral)*i*compound_alt_vol_labels[4] - 0.5*compound_alt_vol_labels[4]

        compound_alt_fe_shift[i,0] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[0] + 0.5*compound_alt_fe_labels[0]
        compound_alt_fe_shift[i,1] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[1] + 0.5*compound_alt_fe_labels[1]
        compound_alt_fe_shift[i,2] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[2] + 0.5*compound_alt_fe_labels[2]
        compound_alt_fe_shift[i,3] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[3] + 0.5*compound_alt_fe_labels[3]
        compound_alt_fe_shift[i,4] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[4] + 0.5*compound_alt_fe_labels[4]
        compound_alt_fe_shift[i,5] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[5] + 0.5*compound_alt_fe_labels[4]


    compound_alt_fe[i,0] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[0]
    compound_alt_fe[i,1] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[1]
    compound_alt_fe[i,2] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[2]
    compound_alt_fe[i,3] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[3]
    compound_alt_fe[i,4] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[4]
    compound_alt_fe[i,5] = 0.78 - (2.7/n_lateral)*i*compound_alt_fe_labels[5]


    for ii in range(len(param_strings)):
        compound_alt_vol_solo[i,ii] = (2.7/n_lateral)*i*value_alt_vol_mean[ii,0,0]
        compound_alt_fe_solo[i,ii] = 0.78 - (2.7/n_lateral)*i*value_alt_fe_mean[ii,0,0]

        if age_vector[i] > 0.5:
            compound_alt_vol_solo_shift[i,ii] = (2.7/n_lateral)*i*value_alt_vol_mean[ii,0,0] - 0.5*value_alt_vol_mean[ii,0,0]
            compound_alt_fe_solo_shift[i,ii] = 0.78 - (2.7/n_lateral)*i*value_alt_fe_mean[ii,0,0] + 0.5*value_alt_fe_mean[ii,0,0]


    # # compound at 10.0 % / Myr
    # if age_vector[i] > 0.5:
    #     compound_alt_vol[i,2] = (2.7/n_lateral)*i*5.2 - 0.5*5.2
    #     compound_alt_fe[i,2] = 0.78 - (2.7/n_lateral)*i*(0.00405) + 0.5*0.00405
    #
    # # compound at 15.0 % / Myr
    # if age_vector[i] > 0.5:
    #     compound_alt_vol[i,3] = (2.7/n_lateral)*i*8.3 - 0.5*8.3
    #     compound_alt_fe[i,3] = 0.78 - (2.7/n_lateral)*i*(0.03188) + 0.5*0.03188
    #
    # # compound SD for alt_fe
    # compound_alt_fe[i,4] = 0.78 - (2.7/n_lateral)*i*(0.06)
    # if age_vector[i] > 0.5:
    #     compound_alt_fe[i,5] = 0.78 - (2.7/n_lateral)*i*(0.06) + 0.5*0.06









#hack: FIG: compounds_lin
print "compounds_lin"
fig=plt.figure(figsize=(12.0,8.0))
rainbow_lw = 1.5

ax=fig.add_subplot(2, 2, 1, frameon=True)

plt.plot(distance_vector,compound_alt_vol[:,0], label=str(compound_alt_vol_labels[0]), c=plot_col[0], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_vol[:,1], label=str(compound_alt_vol_labels[1]), c=plot_col[1], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_vol[:,2], label=str(compound_alt_vol_labels[2]), c=plot_col[2], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_vol[:,3], label=str(compound_alt_vol_labels[3]), c=plot_col[3], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_vol[:,4], label=str(compound_alt_vol_labels[4]), c=plot_col[4], lw=rainbow_lw, zorder=10)

#plt.scatter(site_locations,alt_values,edgecolor=dark_red,color=dark_red,zorder=10,s=60, label="data from sites")
plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)

for j in range(nsites):
    # error bar height
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    # lower error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    # upper error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)

ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

# ax.fill_between(distance_vector, compound_alt_vol[:,2], compound_alt_vol[:,1],zorder=1, facecolor='none', hatch='//', edgecolor='b', lw=0)

plt.xlim([20000.0,110000.0])
plt.ylim([-1.0,30.0])
plt.xlabel("crust age [Myr]")
plt.ylabel("alteration volume percent")
plt.title("compound_alt_vol")
plt.legend(fontsize=8,loc='best',ncol=1)




ax=fig.add_subplot(2, 2, 2, frameon=True)

plt.plot(distance_vector,compound_alt_vol_shift[:,0], label=str(compound_alt_vol_labels[0]), c=plot_col[0], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_vol_shift[:,1], label=str(compound_alt_vol_labels[1]), c=plot_col[1], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_vol_shift[:,2], label=str(compound_alt_vol_labels[2]), c=plot_col[2], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_vol_shift[:,3], label=str(compound_alt_vol_labels[3]), c=plot_col[3], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_vol_shift[:,4], label=str(compound_alt_vol_labels[4]), c=plot_col[4], lw=rainbow_lw, zorder=10)

#plt.scatter(site_locations,alt_values,edgecolor=dark_red,color=dark_red,zorder=10,s=60, label="data from sites")
plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)

for j in range(nsites):
    # error bar height
    plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    # lower error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
    # upper error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)

ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

plt.xlim([20000.0,110000.0])
plt.ylim([-1.0,30.0])
plt.xlabel("crust age [Myr]")
plt.ylabel("alteration volume percent")
plt.title("compound_alt_vol")
plt.legend(fontsize=8,loc='best',ncol=1)




ax=fig.add_subplot(2, 2, 3, frameon=True)

plt.plot(distance_vector,compound_alt_fe[:,0], label=str(compound_alt_fe_labels[0]), c=plot_col[0], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_fe[:,1], label=str(compound_alt_fe_labels[1]), c=plot_col[1], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_fe[:,2], label=str(compound_alt_fe_labels[2]), c=plot_col[2], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_fe[:,3], label=str(compound_alt_fe_labels[3]), c=plot_col[3], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_fe[:,4], label=str(compound_alt_fe_labels[4]), c=plot_col[4], lw=rainbow_lw, zorder=10)

#plt.scatter(site_locations,fe_values,edgecolor=dark_red,color=dark_red,zorder=10,s=60, label="data from sites")
plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')

for j in range(nsites):
    # error bar height
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
    # lower error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    # upper error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)

ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

plt.xlim([20000.0,110000.0])
plt.ylim([0.6,0.8])
plt.xlabel("crust age [Myr]")
plt.ylabel("alteration volume percent")
plt.title("compound_fe_vol")
plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))



ax=fig.add_subplot(2, 2, 4, frameon=True)

plt.plot(distance_vector,compound_alt_fe_shift[:,0], label=str(compound_alt_fe_labels[0]), c=plot_col[0], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_fe_shift[:,1], label=str(compound_alt_fe_labels[1]), c=plot_col[1], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_fe_shift[:,2], label=str(compound_alt_fe_labels[2]), c=plot_col[2], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_fe_shift[:,3], label=str(compound_alt_fe_labels[3]), c=plot_col[3], lw=rainbow_lw, zorder=10)
plt.plot(distance_vector,compound_alt_fe_shift[:,4], label=str(compound_alt_fe_labels[4]), c=plot_col[4], lw=rainbow_lw, zorder=10)

#plt.scatter(site_locations,fe_values,edgecolor=dark_red,color=dark_red,zorder=10,s=60, label="data from sites")
plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')

for j in range(nsites):
    # error bar height
    plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
    # lower error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
    # upper error bar
    plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)

ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)

plt.xlim([20000.0,110000.0])
plt.ylim([0.6,0.8])
plt.xlabel("crust age [Myr]")
plt.ylabel("alteration volume percent")
plt.title("compound_fe_vol")
plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))







plt.savefig(in_path+dir_path+fig_path+"z_compounds_lin.png",bbox_inches='tight')
# plt.savefig(in_path+dir_path+fig_path+"z_compounds_lin.eps",bbox_inches='tight')










#hack: FIG: comp_lin_solo
print "comp_lin_solo"
fig=plt.figure(figsize=(12.0,6.0))
rainbow_lw = 1.5

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
plt.legend(fontsize=8,loc='best',ncol=1)





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
plt.legend(fontsize=8,loc='best',ncol=1)




## RATIOS OF SHIFT

ax=fig.add_subplot(2, 3, 3, frameon=True)

for ii in range(len(param_strings)):
    plt.plot(distance_vector,compound_alt_vol_solo_shift[:,ii]/compound_alt_fe_solo_shift[:,ii], label=str(param_strings[ii]), c=plot_col[ii+1], lw=rainbow_lw, zorder=10)

plt.plot(site_locations, alt_values/fe_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
# for j in range(nsites):
#     plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
#     plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
#     plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
# ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)

plt.xlim([20000.0,110000.0])
#plt.ylim([-1.0,30.0])
# plt.xlabel("crust age [Myr]", fontsize=9)
plt.ylabel("SHIFT RATIOS", fontsize=9)
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




plt.savefig(in_path+dir_path+fig_path+"z_comp_lin_solo.png",bbox_inches='tight')
plt.savefig(in_path+dir_path+fig_path+"zzz_comp_lin_solo.eps",bbox_inches='tight')
# plt.savefig(in_path+dir_path+fig_path+"z_compounds_lin.eps",bbox_inches='tight')






#
#
# #shack: FIG: compounds
# print "compounds"
# fig=plt.figure(figsize=(8.0,8.0))
#
#
# ax=fig.add_subplot(2, 1, 1, frameon=True)
#
# plt.plot(distance_vector,compound_alt_vol[:,0], label='5.2p / Myr', c=plot_col[0], lw=2, zorder=10)
# plt.plot(distance_vector,compound_alt_vol[:,1], label='8.3p / Myr', c=plot_col[1], lw=2, zorder=10)
# plt.plot(distance_vector,compound_alt_vol[:,2], label='5.2p / Myr .5Ma shift', c=plot_col[2], lw=2, zorder=10)
# plt.plot(distance_vector,compound_alt_vol[:,3], label='8.3p / Myr .5Ma shift', c=plot_col[3], lw=2, zorder=10)
#
# #plt.scatter(site_locations,alt_values,edgecolor=dark_red,color=dark_red,zorder=10,s=60, label="data from sites")
# plt.plot(site_locations, alt_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
#
# for j in range(nsites):
#     # error bar height
#     plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
#     # lower error bar
#     plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
#     # upper error bar
#     plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
#
# ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)
#
# ax.fill_between(distance_vector, compound_alt_vol[:,2], compound_alt_vol[:,1],zorder=1, facecolor='none', hatch='//', edgecolor='b', lw=0)
#
# plt.xlim([20000.0,110000.0])
# plt.ylim([-1.0,30.0])
# plt.xlabel("crust age [Myr]")
# plt.ylabel("alteration volume percent")
# plt.title("compound_alt_vol")
# plt.legend(fontsize=8,loc='best',ncol=1)
#
#
#
# ax=fig.add_subplot(2, 1, 2, frameon=True)
#
# plt.plot(distance_vector,compound_alt_fe[:,0], label='0 : -0.00405', c=plot_col[0], lw=2)
# plt.plot(distance_vector,compound_alt_fe[:,1], label='1 : -0.03188', c=plot_col[1], lw=2)
# plt.plot(distance_vector,compound_alt_fe[:,2], label='2 : -0.00405 shift', c=plot_col[2], lw=2)
# plt.plot(distance_vector,compound_alt_fe[:,3], label='3 : -0.03188 shift', c=plot_col[3], lw=2)
#
# plt.plot(distance_vector,compound_alt_fe[:,4], label='4 : -0.06', c=plot_col[4], lw=2)
# plt.plot(distance_vector,compound_alt_fe[:,5], label='5 : -0.06 shift', c=plot_col[5], lw=2)
#
# #plt.scatter(site_locations,fe_values,edgecolor=dark_red,color=dark_red,zorder=10,s=60, label="data from sites")
# plt.plot(site_locations,fe_values,color=dark_red,linestyle='-')
#
# for j in range(nsites):
#     # error bar height
#     plt.plot([site_locations[j],site_locations[j]],[lower_eb_fe[j],upper_eb_fe[j]],c=dark_red, zorder=-1)
#     # lower error bar
#     plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb_fe[j],lower_eb_fe[j]],c=dark_red)
#     # upper error bar
#     plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb_fe[j],upper_eb_fe[j]],c=dark_red)
#
# ax.fill_between(site_locations, lower_eb_fe, upper_eb_fe,zorder=-2, facecolor=fill_color, lw=0)
#
# plt.xlim([20000.0,110000.0])
# plt.ylim([0.6,0.8])
# plt.xlabel("crust age [Myr]")
# plt.ylabel("alteration volume percent")
# plt.title("compound_fe_vol")
# plt.legend(fontsize=8,ncol=2,bbox_to_anchor=(0.5, -0.1))
#
#
#
# plt.savefig(in_path+dir_path+fig_path+"z_compounds.png",bbox_inches='tight')






#todo: contour params
cont_cmap = cm.rainbow
n_cont = 41
cont_skip = 10
bar_shrink = 0.9
clabelpad = 0
xskip = 2
yskip = 1
bar_bins = 4

sp1 = 3
sp2 = 4


all_params = 1

if all_params == 1:
    cont_x_diff_max = len(diff_strings) - 0
    cont_y_param_max = len(param_strings) - 0

x_cont = diff_nums
y_cont = param_nums
x_grid, y_grid = np.meshgrid(x_cont,y_cont)
x_grid = x_grid[:cont_y_param_max,:cont_x_diff_max]
y_grid = y_grid[:cont_y_param_max,:cont_x_diff_max]


#hack: FIG: dsec_contours
print "dsec_contour"
fig=plt.figure(figsize=(12.0,9.0))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

the_min = any_min[0]
the_s = value_dsec[:cont_y_param_max,:cont_x_diff_max,the_min,0]
the_d = value_dsec_d[:cont_y_param_max,:cont_x_diff_max,the_min,0]
the_a = value_dsec_a[:cont_y_param_max,:cont_x_diff_max,the_min,0]
the_b = value_dsec_b[:cont_y_param_max,:cont_x_diff_max,the_min,0]

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

square_contour_min(sp1, sp2, 1, the_s, cb_title="s dsec rate "+secondary[the_min], xlab=1, ylab=1, the_cbar=1, cont_levels_in=cont_levels)

square_contour_min(sp1, sp2, 2, the_d, cb_title="d dsec rate", xlab=1, cont_levels_in=cont_levels)

square_contour_min(sp1, sp2, 3, the_a, cb_title="a dsec rate", xlab=1, cont_levels_in=cont_levels)

square_contour_min(sp1, sp2, 4, the_b, cb_title="b dsec rate", xlab=1, cont_levels_in=cont_levels)

plt.savefig(in_path+dir_path+fig_path+"z_dsec_contours.png",bbox_inches='tight')








#hack: FIG: dsec_pcolor
print "dsec_pcolor"

sp1 = (len(any_min)+1)/2
sp2 = 8


fig=plt.figure(figsize=(20.0,len(any_min)))
plt.subplots_adjust(hspace=0.6)

for j in range(len(any_min)):

    the_min = any_min[j]
    the_s = value_dsec[:cont_y_param_max,:cont_x_diff_max,the_min,0]
    the_d = value_dsec_d[:cont_y_param_max,:cont_x_diff_max,the_min,0]
    the_a = value_dsec_a[:cont_y_param_max,:cont_x_diff_max,the_min,0]
    the_b = value_dsec_b[:cont_y_param_max,:cont_x_diff_max,the_min,0]

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

    if max_all == 0.0:
        max_all = 0.01

    square_pcolor_min(sp1, sp2, sp_factor+1, the_s, cb_title=temp_string + " " + "s dsec rate "+secondary[the_min], xlab=0, ylab=0, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor_min(sp1, sp2, sp_factor+2, the_d, cb_title=" ", xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor_min(sp1, sp2, sp_factor+3, the_a, cb_title=" ", xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor_min(sp1, sp2, sp_factor+4, the_b, cb_title=" ", xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(in_path+dir_path+fig_path+"z_dsec_pcolor.png",bbox_inches='tight')






# #shack: FIG: alt_vol_pcolor
# print "alt_vol_pcolor"
#
# sp1 = 3
# sp2 = 4
#
# cont_x_diff_max = len(diff_strings) - 2
# cont_y_param_max = len(param_strings) - 0
#
#
# fig=plt.figure(figsize=(12.0,9.0))
# plt.subplots_adjust(hspace=0.6)
#
# the_s = value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,0]
# the_d = value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,0]
# the_a = value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,0]
# the_b = value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,0]
#
# min_all = np.min(the_s)
# if np.min(the_d) < min_all:
#     min_all = np.min(the_d)
# if np.min(the_a) < min_all:
#     min_all = np.min(the_a)
# if np.min(the_b) < min_all:
#     min_all = np.min(the_b)
#
# max_all = np.max(the_s)
# if np.max(the_d) > max_all:
#     max_all = np.max(the_d)
# if np.max(the_a) > max_all:
#     max_all = np.max(the_a)
# if np.max(the_b) > max_all:
#     max_all = np.max(the_b)
#
# square_pcolor(sp1, sp2, 1, the_s, cb_title="s alt_vol", xlab=1, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)
#
# square_pcolor(sp1, sp2, 2, the_d, cb_title="d alt_vol", xlab=1, min_all_in=min_all, max_all_in=max_all)
#
# square_pcolor(sp1, sp2, 3, the_a, cb_title="a alt_vol", xlab=1, min_all_in=min_all, max_all_in=max_all)
#
# square_pcolor(sp1, sp2, 4, the_b, cb_title="b alt_vol", xlab=1, min_all_in=min_all, max_all_in=max_all)
#
#
#
# the_s = value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,0]
# the_d = value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,0]
# the_a = value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,0]
# the_b = value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,0]
#
# min_all = np.min(the_s)
# if np.min(the_d) < min_all:
#     min_all = np.min(the_d)
# if np.min(the_a) < min_all:
#     min_all = np.min(the_a)
# if np.min(the_b) < min_all:
#     min_all = np.min(the_b)
#
# max_all = np.max(the_s)
# if np.max(the_d) > max_all:
#     max_all = np.max(the_d)
# if np.max(the_a) > max_all:
#     max_all = np.max(the_a)
# if np.max(the_b) > max_all:
#     max_all = np.max(the_b)
#
# square_pcolor(sp1, sp2, 5, the_s, cb_title="s alt_fe", xlab=1, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)
#
# square_pcolor(sp1, sp2, 6, the_d, cb_title="d alt_fe", xlab=1, min_all_in=min_all, max_all_in=max_all)
#
# square_pcolor(sp1, sp2, 7, the_a, cb_title="a alt_fe", xlab=1, min_all_in=min_all, max_all_in=max_all)
#
# square_pcolor(sp1, sp2, 8, the_b, cb_title="b alt_fe", xlab=1, min_all_in=min_all, max_all_in=max_all)
#
#
# plt.savefig(in_path+dir_path+fig_path+"z_alt_vol_pcolor.png",bbox_inches='tight')






#hack: FIG: alt_vol_pcolor_full
print "alt_vol_pcolor_full"

sp1 = 4
sp2 = 4

cont_x_diff_max = len(diff_strings) - 0
cont_y_param_max = len(param_strings) - 0


fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)



the_s = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,0])
the_d = np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,0])
the_a = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,0])
the_b = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,0])

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

# min_all = 1.0
# max_all = 17.0

square_pcolor(sp1, sp2, 1, the_s, cb_title=temp_string + " " + "s dpri", xlab=1, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 2, the_d, cb_title=temp_string + " " + "d dpri", xlab=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 3, the_a, cb_title=temp_string + " " + "a dpri", xlab=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 4, the_b, cb_title=temp_string + " " + "b dpri", xlab=1, min_all_in=min_all, max_all_in=max_all)






the_s = value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,0]
the_d = value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,0]
the_a = value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,0]
the_b = value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,0]

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

# min_all = 1.0
# max_all = 17.0

square_pcolor(sp1, sp2, 5, the_s, cb_title=temp_string + " " + "s alt_vol", xlab=1, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 6, the_d, cb_title=temp_string + " " + "d alt_vol", xlab=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 7, the_a, cb_title=temp_string + " " + "a alt_vol", xlab=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 8, the_b, cb_title=temp_string + " " + "b alt_vol", xlab=1, min_all_in=min_all, max_all_in=max_all)



the_s = value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,0]
the_d = value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,0]
the_a = value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,0]
the_b = value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,0]

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

# min_all = 0.005
# max_all = 0.10

square_pcolor(sp1, sp2, 9, the_s, cb_title=temp_string + " " + "s alt_fe", xlab=1, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 10, the_d, cb_title=temp_string + " " + "d alt_fe", xlab=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 11, the_a, cb_title=temp_string + " " + "a alt_fe", xlab=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 12, the_b, cb_title=temp_string + " " + "b alt_fe", xlab=1, min_all_in=min_all, max_all_in=max_all)












the_s = value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,0]
the_d = value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,0]
the_a = value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,0]
the_b = value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,0]

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

# min_all = 0.005
# max_all = 0.10

square_pcolor(sp1, sp2, 13, the_s, cb_title=temp_string + " " + "s slope ratio", xlab=1, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 14, the_d, cb_title=temp_string + " " + "d slope ratio", xlab=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 15, the_a, cb_title=temp_string + " " + "a slope ratio", xlab=1, min_all_in=min_all, max_all_in=max_all)

square_pcolor(sp1, sp2, 16, the_b, cb_title=temp_string + " " + "b slope ratio", xlab=1, min_all_in=min_all, max_all_in=max_all)


plt.savefig(in_path+dir_path+fig_path+"z_alt_vol_pcolor_full.png",bbox_inches='tight')
