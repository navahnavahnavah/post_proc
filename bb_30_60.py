# bb_30_60.py

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
temp_string = "60"
# temp_string_list = ['20', '30', '40', '50', '60']
temp_string_list = ['30', '30', '40', '50', '60']
in_path = "../output/revival/winter_basalt_box/"




param_strings = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']
param_nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# diff_strings = ['2.00', '2.50', '3.00', '3.50', '4.00', '4.50', '5.00', '5.50', '6.00']
# diff_nums = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

diff_strings = ['2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50', '3.75', '4.00', '4.25', '4.50', '4.75', '5.00', '5.25', '5.50', '5.75', '6.00']
diff_nums = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 4.25, 5.5, 5.75, 6.0]

#poop: make 2d alt_ind grids
n_grids = 5


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


    for i in range(n_lateral-1):


        #hack: define the shift!!
        shift_myr = 0.5
        active_myr = 3.5
        for ii in range(len(param_strings)):
            compound_alt_vol_solo[i,ii,iii] = (active_myr/n_lateral)*i*value_alt_vol_mean[ii,0,iii]
            compound_alt_fe_solo[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*value_alt_fe_mean[ii,0,iii]

            compound_alt_vol_dual_max[i,ii,iii] = (active_myr/n_lateral)*i*np.max(value_alt_vol_mean_d[:,:,iii])
            compound_alt_fe_dual_max[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.max(value_alt_fe_mean_d[:,:,iii])

            compound_alt_vol_dual_min[i,ii,iii] = (active_myr/n_lateral)*i*np.min(value_alt_vol_mean_d[:,:,iii])
            compound_alt_fe_dual_min[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.min(value_alt_fe_mean_d[:,:,iii])

            if age_vector[i] > shift_myr:
                compound_alt_vol_solo_shift[i,ii,iii] = (active_myr/n_lateral)*i*value_alt_vol_mean[ii,0,iii] - shift_myr*value_alt_vol_mean[ii,0,iii]
                compound_alt_fe_solo_shift[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*value_alt_fe_mean[ii,0,iii] + shift_myr*value_alt_fe_mean[ii,0,iii]

                compound_alt_vol_dual_max_shift[i,ii,iii] = (active_myr/n_lateral)*i*np.max(value_alt_vol_mean_d[:,:,iii]) - shift_myr*np.max(value_alt_vol_mean_d[:,:,iii])
                compound_alt_fe_dual_max_shift[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.max(value_alt_fe_mean_d[:,:,iii]) + shift_myr*np.max(value_alt_fe_mean_d[:,:,iii])

                compound_alt_vol_dual_min_shift[i,ii,iii] = (active_myr/n_lateral)*i*np.min(value_alt_vol_mean_d[:,:,iii]) - shift_myr*np.min(value_alt_vol_mean_d[:,:,iii])
                compound_alt_fe_dual_min_shift[i,ii,iii] = 0.78 - (active_myr/n_lateral)*i*np.min(value_alt_fe_mean_d[:,:,iii]) + shift_myr*np.min(value_alt_fe_mean_d[:,:,iii])











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




## RATIOS OF SHIFT

# ax=fig.add_subplot(2, 3, 3, frameon=True)
#
# for ii in range(len(param_strings)):
#     plt.plot(distance_vector,compound_alt_vol_solo_shift[:,ii]/compound_alt_fe_solo_shift[:,ii], label=str(param_strings[ii]), c=plot_col[ii+1], lw=rainbow_lw, zorder=10)
#
# plt.plot(site_locations, alt_values/fe_values, color=dark_red, linestyle='-', lw=data_lw, zorder=3)
# # for j in range(nsites):
# #     plt.plot([site_locations[j],site_locations[j]],[lower_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
# #     plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[lower_eb[j],lower_eb[j]],c=dark_red, lw=data_lw, zorder=3)
# #     plt.plot([site_locations[j]-ebw,site_locations[j]+ebw],[upper_eb[j],upper_eb[j]],c=dark_red, lw=data_lw, zorder=3)
# # ax.fill_between(site_locations, lower_eb, upper_eb, facecolor=fill_color, lw=0, zorder=0)
#
# plt.xlim([20000.0,110000.0])
# #plt.ylim([-1.0,30.0])
# # plt.xlabel("crust age [Myr]", fontsize=9)
# plt.ylabel("SHIFT RATIOS", fontsize=9)
# #plt.legend(fontsize=8,loc='best',ncol=1)






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




plt.savefig(in_path+dir_path+fig_path+"z_comp_lin_solo_36.png",bbox_inches='tight')
plt.savefig(in_path+dir_path+fig_path+"zzz_comp_lin_solo_36.eps",bbox_inches='tight')
# plt.savefig(in_path+dir_path+fig_path+"z_compounds_lin.eps",bbox_inches='tight')
















dual_alpha = 0.4
hatch_alpha = 1.0
hatch_string = '||'
hatch_strings = ['\\\\', '\\\\', '\\\\', '\\\\', '\\\\']
#hatch_strings = ['\\\\', 'o', '//', '++', '..']
hatch_strings = ['\\\\', '\\\\', '.', '\\\\', '||']
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

    plt.plot(distance_vector, compound_alt_vol_solo[:,0,iii], lw=lw_for_plot, color=plot_col[iii+1])
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

    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,0,iii], lw=lw_for_plot, color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,-1,iii], lw=lw_for_plot, color=plot_col[iii+1])

print "compare minimum dual 60"
print compound_alt_vol_dual_min_shift[:,0,-1]
print "compare minimum solo 50"
print compound_alt_vol_solo_shift[:,0,-2]

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














hatch_strings = ['//', '//', '//', '//', '//']



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




plt.savefig(in_path+dir_path+fig_path+"z_comp_lin_fill_36.png",bbox_inches='tight')
plt.savefig(in_path+dir_path+fig_path+"z_comp_lin_fill_36.eps",bbox_inches='tight')








##hack: temp shift!
compound_alt_vol_solo = compound_alt_vol_solo_shift
compound_alt_fe_solo = compound_alt_fe_solo_shift






#hack: FIG: comp_lin_INIT_36
print "comp_lin_INIT_36"
fig=plt.figure(figsize=(23.0,8.0))
rainbow_lw = 1.0










ax=fig.add_subplot(2, 5, 1, frameon=True)

for iii in range(len(temp_string_list)):

    plt.plot(distance_vector, compound_alt_vol_solo[:,0,iii]/(0.78001-compound_alt_fe_solo[:,0,iii]), color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii]/(0.78001-compound_alt_fe_solo[:,-1,iii]), color=plot_col[iii+1])

plt.plot(site_locations, alt_values/(0.78-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.8-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.82-fe_values), color='k', linestyle='-', lw=3, zorder=13)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.0,300.0])
plt.ylabel("ratio", fontsize=9)
plt.title('ratio solo/solo 0.78, (((0.78, 0.80, 0.82)))')





ax=fig.add_subplot(2, 5, 2, frameon=True)

for iii in range(len(temp_string_list)):

    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,0,iii]/(0.79-compound_alt_fe_solo_shift[:,0,iii]), color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,-1,iii]/(0.79-compound_alt_fe_solo_shift[:,-1,iii]), color=plot_col[iii+1])

plt.plot(site_locations, alt_values/(0.78-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.8-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.82-fe_values), color='k', linestyle='-', lw=3, zorder=13)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.0,300.0])
plt.ylabel("ratio", fontsize=9)
plt.title('ratio solo/solo 0.79, (((0.78, 0.80, 0.82)))')







ax=fig.add_subplot(2, 5, 3, frameon=True)

for iii in range(len(temp_string_list)):

    plt.plot(distance_vector, compound_alt_vol_solo[:,0,iii]/(0.80-compound_alt_fe_solo[:,0,iii]), color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii]/(0.80-compound_alt_fe_solo[:,-1,iii]), color=plot_col[iii+1])

plt.plot(site_locations, alt_values/(0.78-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.8-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.82-fe_values), color='k', linestyle='-', lw=3, zorder=13)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.0,300.0])
plt.ylabel("ratio", fontsize=9)
plt.title('ratio solo/solo 0.80, (((0.78, 0.80, 0.82)))')




ax=fig.add_subplot(2, 5, 4, frameon=True)

for iii in range(len(temp_string_list)):

    plt.plot(distance_vector, compound_alt_vol_solo[:,0,iii]/(0.81-compound_alt_fe_solo[:,0,iii]), color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii]/(0.81-compound_alt_fe_solo[:,-1,iii]), color=plot_col[iii+1])

plt.plot(site_locations, alt_values/(0.78-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.8-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.82-fe_values), color='k', linestyle='-', lw=3, zorder=13)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.0,300.0])
plt.ylabel("ratio", fontsize=9)
plt.title('ratio solo/solo 0.81, (((0.78, 0.80, 0.82)))')





ax=fig.add_subplot(2, 5, 5, frameon=True)

for iii in range(len(temp_string_list)):

    plt.plot(distance_vector, compound_alt_vol_solo[:,0,iii]/(0.82-compound_alt_fe_solo[:,0,iii]), color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii]/(0.82-compound_alt_fe_solo[:,-1,iii]), color=plot_col[iii+1])

plt.plot(site_locations, alt_values/(0.78-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.8-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.82-fe_values), color='k', linestyle='-', lw=3, zorder=13)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.0,300.0])
plt.ylabel("ratio", fontsize=9)
plt.title('ratio solo/solo 0.82, (((0.78, 0.80, 0.82)))')






ax=fig.add_subplot(2, 5, 6, frameon=True)

for iii in range(len(temp_string_list)):

    plt.plot(distance_vector, compound_alt_vol_solo[:,0,iii]/(0.83-compound_alt_fe_solo[:,0,iii]), color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii]/(0.83-compound_alt_fe_solo[:,-1,iii]), color=plot_col[iii+1])

plt.plot(site_locations, alt_values/(0.78-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.8-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.82-fe_values), color='k', linestyle='-', lw=3, zorder=13)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.0,300.0])
plt.ylabel("ratio", fontsize=9)
plt.title('ratio solo/solo 0.83, (((0.78, 0.80, 0.82)))')







ax=fig.add_subplot(2, 5, 7, frameon=True)

for iii in range(len(temp_string_list)):

    plt.plot(distance_vector, compound_alt_vol_solo[:,0,iii]/(0.88-compound_alt_fe_solo[:,0,iii]), color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_vol_solo[:,-1,iii]/(0.88-compound_alt_fe_solo[:,-1,iii]), color=plot_col[iii+1])

plt.plot(site_locations, alt_values/(0.78-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.8-fe_values), color='k', linestyle='-', lw=3, zorder=13)
plt.plot(site_locations, alt_values/(0.82-fe_values), color='k', linestyle='-', lw=3, zorder=13)

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.0,300.0])
plt.ylabel("ratio", fontsize=9)
plt.title('ratio solo/solo 0.88, (((0.78, 0.80, 0.82)))')




ax=fig.add_subplot(2, 5, 10, frameon=True)

for iii in range(len(temp_string_list)):

    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,0,iii]/(0.78-compound_alt_fe_solo[:,0,iii]), color=plot_col[iii+1])
    plt.plot(distance_vector, compound_alt_vol_solo_shift[:,-1,iii]/(0.78-compound_alt_fe_solo[:,-1,iii]), color=plot_col[iii+1])

plt.plot(site_locations, alt_values/(0.78-fe_values), color='k', linestyle='-', lw=3, zorder=13)

ratio_upper = np.zeros(len(upper_eb_fe))
ratio_lower = np.zeros(len(upper_eb_fe))

for iii in range(len(ratio_upper)):
    ratio_upper[iii] = np.max([upper_eb[iii]/(0.78-upper_eb_fe[iii]) , upper_eb[iii]/(0.78-lower_eb_fe[iii]) , lower_eb[iii]/(0.78-upper_eb_fe[iii]) , lower_eb[iii]/(0.78-lower_eb_fe[iii]) ])

    ratio_lower[iii] = np.min([upper_eb[iii]/(0.78-upper_eb_fe[iii]) , upper_eb[iii]/(0.78-lower_eb_fe[iii]) , lower_eb[iii]/(0.78-upper_eb_fe[iii]) , lower_eb[iii]/(0.78-lower_eb_fe[iii]) ])

print "debug upper"
print ratio_upper
print " "
print "debug lower"
print ratio_lower
print " "
print " "

ax.fill_between(site_locations, ratio_lower, ratio_upper, facecolor=fill_color, lw=0, zorder=0)

plt.plot(site_locations, ratio_lower, color='k')
plt.plot(site_locations, ratio_upper, color='c')

plt.xlim([20000.0,110000.0])
plt.xticks(np.linspace(20000,110000,10), np.linspace(2,11,10))
plt.ylim([0.0,300.0])
plt.ylabel("ratio", fontsize=9)
plt.title('ratio shift/solo, 0.78')



plt.savefig(in_path+dir_path+fig_path+"z_comp_lin_INIT_36.png",bbox_inches='tight')
#plt.savefig(in_path+dir_path+fig_path+"z_comp_lin_fill_36.eps",bbox_inches='tight')





sp1 = 4
sp2 = len(temp_string_list)
cont_cmap = cm.rainbow
xskip = 2
yskip = 1
bar_bins = 4

cont_x_diff_max = len(diff_strings) - 0
cont_y_param_max = len(param_strings) - 0




#hack: FIG: ai_pri_36
print "ai_pri_36"

fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

#value_alt_vol_mean_d[:,:,iii]

for iii in range(len(temp_string_list)):

    the_s = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,:])

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

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s dpri " + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d dpri " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a dpri " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b dpri " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)




#
#
# the_s = value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,0]
# the_d = value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,0]
# the_a = value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,0]
# the_b = value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,0]
#
# # the_s = value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,0]
# # the_d = value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,0]
# # the_a = value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,0]
# # the_b = value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,0]/value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,0]
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
# # min_all = 0.005
# # max_all = 0.10
#
# square_pcolor(sp1, sp2, 13, the_s, cb_title=temp_string + " " + "s slope ratio", xlab=1, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)
#
# square_pcolor(sp1, sp2, 14, the_d, cb_title=temp_string + " " + "d slope ratio", xlab=1, min_all_in=min_all, max_all_in=max_all)
#
# square_pcolor(sp1, sp2, 15, the_a, cb_title=temp_string + " " + "a slope ratio", xlab=1, min_all_in=min_all, max_all_in=max_all)
#
# square_pcolor(sp1, sp2, 16, the_b, cb_title=temp_string + " " + "b slope ratio", xlab=1, min_all_in=min_all, max_all_in=max_all)


plt.savefig(in_path+dir_path+fig_path+"z_ai_pri_36.png",bbox_inches='tight')








#hack: FIG: ai_1_36
print "ai_1_36"

fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    the_s = np.abs(value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,:])

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

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s alt_vol " + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d alt_vol " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a alt_vol " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b alt_vol " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(in_path+dir_path+fig_path+"z_ai_1_36.png",bbox_inches='tight')






#hack: FIG: ai_2_36
print "ai_2_36"

fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    the_s = np.abs(value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,:])

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

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s alt_fe " + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d alt_fe " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a alt_fe " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b alt_fe " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(in_path+dir_path+fig_path+"z_ai_2_36.png",bbox_inches='tight')







#hack: FIG: ai_3_36
print "ai_3_36"
# new ratio is d_pri : alt-fe

fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    # the_s = np.abs(value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    # the_d = np.abs(value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    # the_a = np.abs(value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    # the_b = np.abs(value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])
    #
    # the_s_row = np.abs(value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,:])
    # the_d_row = np.abs(value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    # the_a_row = np.abs(value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    # the_b_row = np.abs(value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,:])

    # the_s = np.abs(value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,iii]/value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    # the_d = np.abs(value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,iii]/value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    # the_a = np.abs(value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,iii]/value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    # the_b = np.abs(value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,iii]/value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])
    #
    # the_s_row = np.abs(value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,:]/value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,:])
    # the_d_row = np.abs(value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,:]/value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    # the_a_row = np.abs(value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,:]/value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    # the_b_row = np.abs(value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,:]/value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,:])

    the_s = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = the_a

    the_s_row = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,:])
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

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s RATIO " + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d RATIO " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a RATIO " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b RATIO " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(in_path+dir_path+fig_path+"z_ai_3_36.png",bbox_inches='tight')









#hack: FIG: ai_4_36
print "ai_4_36"
# new ratio is  alt-fe : dpri

fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):


    the_s = np.abs(value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,iii]/value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,iii]/value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,iii]/value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,iii]/value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = the_a

    the_s_row = np.abs(value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,:]/value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,:]/value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,:]/value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,:]/value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,:])
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

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s RATIO FLIP " + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d RATIO flip " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a RATIO flip " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b RATIO flip " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(in_path+dir_path+fig_path+"z_ai_4_36.png",bbox_inches='tight')










#hack: FIG: ai_5_36
print "ai_5_36"
# new ratio is d_pri : alt-fe

fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    the_s = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,iii]/value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = the_a

    the_s_row = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,:]/value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,:])
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

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s RATIO " + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d RATIO " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a RATIO " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)
    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b RATIO " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(in_path+dir_path+fig_path+"z_ai_5_36.png",bbox_inches='tight')











#poop: FIG: ai_CHUNKS_36
print "ai_CHUNKS_36"
# new ratio is d_pri : alt-fe

sp11 = 4
sp22 = len(temp_string_list)

fig=plt.figure(figsize=(12.0,12.0))
#plt.subplots_adjust(hspace=0.4)

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

    ax=fig.add_subplot(sp22, sp11, iii*sp11 + 1, frameon=True)
    for ii in range(len(param_strings)):
        # red = alt_fe_mean_d
        # plt.plot(diff_nums,value_alt_fe_mean[ii,:,iii]/np.max(value_alt_fe_mean[ii,:,iii]),color='r',zorder=3)
        plt.plot(diff_nums,value_alt_fe_mean[ii,:,iii]/max_all_alt_fe,color='r',zorder=3)
        # blue = dpri
        # plt.plot(diff_nums,np.abs(value_dpri_mean_s[ii,:,iii])/np.max(np.abs(value_dpri_mean_s[ii,:,iii])),color='b',zorder=5)
        plt.plot(diff_nums,np.abs(value_dpri_mean[ii,:,iii])/max_all_dpri_mean,color='b',zorder=5)
    plt.title('temp = ' + temp_string_list[iii] + 'chamber s')
    plt.ylim([min_all,max_all])
    if iii != len(temp_string_list)-1:
        plt.xticks([])



    ax=fig.add_subplot(sp22, sp11, iii*sp11 + 2, frameon=True)
    for ii in range(len(param_strings)):
        # red = alt_fe_mean_d
        # plt.plot(diff_nums,value_alt_fe_mean_d[ii,:,iii]/np.max(value_alt_fe_mean_d[ii,:,iii]),color='r',zorder=3)
        plt.plot(diff_nums,value_alt_fe_mean_d[ii,:,iii]/max_all_alt_fe,color='r',zorder=3)
        # blue = dpri
        plt.plot(diff_nums,np.abs(value_dpri_mean_d[ii,:,iii])/max_all_dpri_mean,color='b',zorder=5)
    plt.title('temp = ' + temp_string_list[iii] + 'chamber d')
    plt.ylim([min_all,max_all])
    if iii != len(temp_string_list)-1:
        plt.xticks([])

    ax=fig.add_subplot(sp22, sp11, iii*sp11 + 3, frameon=True)
    for ii in range(len(param_strings)):
        # red = alt_fe_mean_d
        # plt.plot(diff_nums,value_alt_fe_mean_a[ii,:,iii]/np.max(value_alt_fe_mean_a[ii,:,iii]),color='r',zorder=3)
        plt.plot(diff_nums,value_alt_fe_mean_a[ii,:,iii]/max_all_alt_fe,color='r',zorder=3)
        # blue = dpri
        plt.plot(diff_nums,np.abs(value_dpri_mean_a[ii,:,iii])/max_all_dpri_mean,color='b',zorder=5)
    plt.title('temp = ' + temp_string_list[iii] + 'chamber a')
    plt.ylim([min_all,max_all])
    if iii != len(temp_string_list)-1:
        plt.xticks([])

    ax=fig.add_subplot(sp22, sp11, iii*sp11 + 4, frameon=True)
    for ii in range(len(param_strings)):
        # red = alt_fe_mean_d
        # plt.plot(diff_nums,value_alt_fe_mean_b[ii,:,iii]/np.max(value_alt_fe_mean_b[ii,:,iii]),color='r',zorder=3)
        plt.plot(diff_nums,value_alt_fe_mean_b[ii,:,iii]/max_all_alt_fe,color='r',zorder=3)
        # blue = dpri
        plt.plot(diff_nums,np.abs(value_dpri_mean_b[ii,:,iii])/max_all_dpri_mean,color='b',zorder=5)
    plt.title('temp = ' + temp_string_list[iii] + 'chamber b')
    plt.ylim([min_all,max_all])
    if iii != len(temp_string_list)-1:
        plt.xticks([])




plt.savefig(in_path+dir_path+fig_path+"z_ai_CHUNKS_36.png",bbox_inches='tight')









#hack: FIG: ai_dpri_to_feot
print "ai_dpri_to_feot"
save_string = "dpri_to_feot "


fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    the_s = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,iii]/save_feot_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = 2.0*np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,iii]/save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,iii]/save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,iii]/save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = the_a

    the_s_row = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,:]/save_feot_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = 2.0*np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,:]/save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,:]/save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,:]/save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,:])
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

plt.savefig(in_path+dir_path+fig_path+"z_ai_dpri_to_feot.png",bbox_inches='tight')








#hack: FIG: ai_dpri_to_mgo
print "ai_dpri_to_mgo"
save_string = "dpri_to_mgo "


fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    the_s = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,iii]/save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = 2.0*np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,iii]/save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,iii]/save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,iii]/save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = the_a

    the_s_row = np.abs(value_dpri_mean[:cont_y_param_max,:cont_x_diff_max,:]/save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = 2.0*np.abs(value_dpri_mean_d[:cont_y_param_max,:cont_x_diff_max,:]/save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(value_dpri_mean_a[:cont_y_param_max,:cont_x_diff_max,:]/save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(value_dpri_mean_b[:cont_y_param_max,:cont_x_diff_max,:]/save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,:])
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

plt.savefig(in_path+dir_path+fig_path+"z_ai_dpri_to_mgo.png",bbox_inches='tight')







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

plt.savefig(in_path+dir_path+fig_path+"z_ai_calc_to_dpri.png",bbox_inches='tight')







#poop: THE 2D SAVE JUNE FIGS!

#hack: FIG: save_feot
print "save_feot"

fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    the_s = np.abs(save_feot_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(save_feot_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,:])

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

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s save_feot " + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d save_feot " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a save_feot " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b save_feot " + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(in_path+dir_path+fig_path+"save_feot.png",bbox_inches='tight')







#hack: FIG: save_fe_num
print "save_fe_num"
save_string = "save_fe_num "

fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    the_s = np.abs(save_feot_mean[:cont_y_param_max,:cont_x_diff_max,iii]) / (save_feot_mean[:cont_y_param_max,:cont_x_diff_max,iii] + save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,iii]) / (save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,iii] + save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,iii]) / (save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,iii] + save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,iii]) / (save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,iii] + save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(save_feot_mean[:cont_y_param_max,:cont_x_diff_max,:]) / (save_feot_mean[:cont_y_param_max,:cont_x_diff_max,:] + save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,:]) / (save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,:] + save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,:]) / (save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,:] + save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,:]) / (save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,:] + save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,:])

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

plt.savefig(in_path+dir_path+fig_path+"save_fe_num.png",bbox_inches='tight')












#hack: FIG: save_mg_num
print "save_mg_num"
save_string = "save_mg_num "

fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    the_s = np.abs(save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,iii]) / (save_feot_mean[:cont_y_param_max,:cont_x_diff_max,iii] + save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,iii]) / (save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,iii] + save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,iii]) / (save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,iii] + save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,iii]) / (save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,iii] + save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,:]) / (save_feot_mean[:cont_y_param_max,:cont_x_diff_max,:] + save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,:]) / (save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,:] + save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,:]) / (save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,:] + save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,:]) / (save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,:] + save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,:])

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

plt.savefig(in_path+dir_path+fig_path+"save_mg_num.png",bbox_inches='tight')









#hack: FIG: save_mg_to_fe
print "save_mg_to_fe"
save_string = "mg_to_fe"

fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    the_s = np.abs(save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,iii]) / ( save_feot_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,iii]) / ( save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,iii]) / ( save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,iii]) / ( save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(save_mgo_mean[:cont_y_param_max,:cont_x_diff_max,:]) / ( save_feot_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(save_mgo_mean_d[:cont_y_param_max,:cont_x_diff_max,:]) / ( save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(save_mgo_mean_a[:cont_y_param_max,:cont_x_diff_max,:]) / ( save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,:]) / ( save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,:])

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

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s " + save_string + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(in_path+dir_path+fig_path+"save_mg_to_fe.png",bbox_inches='tight')









#hack: FIG: save_k_to_fe
print "save_k_to_fe"
save_string = "k_to_fe"

fig=plt.figure(figsize=(8.0,8.0))
plt.subplots_adjust(hspace=0.4)

for iii in range(len(temp_string_list)):

    the_s = np.abs(save_k2o_mean[:cont_y_param_max,:cont_x_diff_max,iii]) / ( save_feot_mean[:cont_y_param_max,:cont_x_diff_max,iii])
    the_d = np.abs(save_k2o_mean_d[:cont_y_param_max,:cont_x_diff_max,iii]) / ( save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,iii])
    the_a = np.abs(save_k2o_mean_a[:cont_y_param_max,:cont_x_diff_max,iii]) / ( save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,iii])
    the_b = np.abs(save_mgo_mean_b[:cont_y_param_max,:cont_x_diff_max,iii]) / ( save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,iii])

    the_s_row = np.abs(save_k2o_mean[:cont_y_param_max,:cont_x_diff_max,:]) / ( save_feot_mean[:cont_y_param_max,:cont_x_diff_max,:])
    the_d_row = np.abs(save_k2o_mean_d[:cont_y_param_max,:cont_x_diff_max,:]) / ( save_feot_mean_d[:cont_y_param_max,:cont_x_diff_max,:])
    the_a_row = np.abs(save_k2o_mean_a[:cont_y_param_max,:cont_x_diff_max,:]) / ( save_feot_mean_a[:cont_y_param_max,:cont_x_diff_max,:])
    the_b_row = np.abs(save_k2o_mean_b[:cont_y_param_max,:cont_x_diff_max,:]) / ( save_feot_mean_b[:cont_y_param_max,:cont_x_diff_max,:])

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

    square_pcolor(sp2, sp1, iii*sp1 + 1, the_s, cb_title="s " + save_string + temp_string_list[iii], xlab=0, ylab=1, the_cbar=1, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 2, the_d, cb_title="d " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 3, the_a, cb_title="a " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

    square_pcolor(sp2, sp1, iii*sp1 + 4, the_b, cb_title="b " + save_string + temp_string_list[iii], xlab=0, min_all_in=min_all, max_all_in=max_all)

plt.savefig(in_path+dir_path+fig_path+"save_k_to_fe.png",bbox_inches='tight')
