# flow_history.py

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

plot_col = ['#000000', '#940000', '#d26618', '#dfa524', '#9ac116', '#139a31', '#35b5aa', '#0740d2', '#7f05d4', '#b100de']

col = ['#6e0202', '#fc385b', '#ff7411', '#19a702', '#00520d', '#00ffc2', '#609ff2', '#20267c','#8f00ff', '#ec52ff', '#6e6e6e', '#000000', '#df9a00', '#7d4e22', '#ffff00', '#df9a00', '#812700', '#6b3f67', '#0f9995', '#4d4d4d', '#d9d9d9', '#e9acff']

outpath = "../output/revival/winter_basalt_box/"


def square_contour(sp1, sp2, sp, cont_block, cb_title="", xlab=0, ylab=0, age_ticks=[1.0, 2.0], age_tick_labels=[1.0, 2.0]):
    ax1=fig.add_subplot(sp1, sp2, sp, frameon=True)

    if xlab == 1:
        plt.xlabel('log10(mixing time [years])', fontsize=9)
    if ylab == 1:
        plt.ylabel('discharge q [m/yr]', fontsize=9)

    pCont = ax1.contourf(x_grid,y_grid,cont_block, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
    for c in pCont.collections:
        c.set_edgecolor("face")

    # plt.contour(x_grid, y_grid, cont_block, levels=alt_vol_data_contours, colors='w', linewidth=3.0, zindex=20)
    # plt.contour(x_grid, y_grid, cont_block, levels=[0.0, 4.7, 5.38, 10.2], colors=('y', 'k', '#656565', 'w'), linewidth=3.0, zindex=20)
    plt.contour(x_grid, y_grid, cont_block, levels=[0.0, 4.7, 5.38, 10.2], colors='w', linewidth=6.0)



    plt.xticks(diff_nums[:cont_x_diff_max:xskip],diff_strings[::xskip], fontsize=8)
    plt.yticks(param_nums[:cont_y_param_max:yskip],param_strings[::yskip], fontsize=8)

    ax2 = ax1.twinx()
    plt.yticks(age_tick_labels, age_ticks, fontsize=8)
    plt.ylim([0.5,4.5])

    bbox = ax2.get_position()
    cax = fig.add_axes([bbox.xmin+0.0, bbox.ymin-0.05, bbox.width*1.0, bbox.height*0.05])
    cbar = plt.colorbar(pCont, cax = cax,orientation='horizontal')
    cbar.set_ticks(cont_levels[::cont_skip])
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_xlabel(cb_title,fontsize=9,labelpad=clabelpad)
    cbar.solids.set_edgecolor("face")

    return square_contour

def square_contour_min(sp1, sp2, sp, cont_block, cb_title="", xlab=0, ylab=0, age_ticks=[1.0, 2.0], age_tick_labels=[1.0, 2.0], the_cbar=0):
    ax1=fig.add_subplot(sp1, sp2, sp, frameon=True)

    if xlab == 1:
        plt.xlabel('log10(mixing time [years])', fontsize=8)
    if ylab == 1:
        plt.ylabel('discharge q [m/yr]', fontsize=8)

    pCont = ax1.contourf(x_grid,y_grid,cont_block, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
    for c in pCont.collections:
        c.set_edgecolor("face")




    plt.xticks(diff_nums[:cont_x_diff_max:xskip],diff_strings[::xskip], fontsize=8)
    plt.yticks(param_nums[:cont_y_param_max:yskip],param_strings[::yskip], fontsize=8)

    ax2 = ax1.twinx()
    plt.yticks(age_tick_labels, age_ticks, fontsize=8)
    plt.ylim([0.5,4.5])

    plt.title(cb_title, fontsize=9)

    if the_cbar == 1:
        bbox = ax2.get_position()
        cax = fig.add_axes([bbox.xmin+0.25, bbox.ymin-0.05, bbox.width*2.0, bbox.height*0.04])
        cbar = plt.colorbar(pCont, cax = cax,orientation='horizontal')
        cbar.set_ticks(cont_levels[::cont_skip])
        cbar.ax.tick_params(labelsize=7)
        # cbar.ax.set_xlabel(cb_title,fontsize=9,labelpad=clabelpad)
        cbar.solids.set_edgecolor("face")

    return square_contour_min



def any_2d_interp(x_in, y_in, z_in, x_diff_path, y_param_path, kind_in='linear'):

    the_f = interpolate.interp2d(x_in, y_in, z_in, kind=kind_in)
    any_2d_interp = the_f(x_diff_path, y_param_path)

    return any_2d_interp


# def any_2d_interp_pt(x_in, y_in, z_in, x_diff_path, y_param_path, kind_in='linear'):
#
#     the_f = interpolate.interp2d(x_in, y_in, z_in, kind=kind_in)
#     any_2d_interp_tp = the_f_pt(x_diff_pt, y_param_pt)
#
#     return any_2d_interp_tp






nsteps = 1000
n_hs = 3
n_dx = 3
n_hist = 10
n_hist_active = 2

param_c = 4200.0
param_lambda = 1.2
param_hs = 5.0 # sedimentation rate 5m/Myr
param_hb = np.zeros(n_hist) # aquifer thickness 500m
param_rho = 1000.0
param_dx = 50000.0

steps = np.arange(nsteps)
age_vec = np.linspace(0.0, 5.0, nsteps)

hs_vec = np.zeros([len(steps),n_hist])
dx_vec = np.zeros([len(steps),n_hist])

q_vec = np.zeros([len(steps),n_hist])
q_vec_log10 = np.zeros([len(steps),n_hist])
q_vec_log10_myr = np.zeros([len(steps),n_hist])


# txt_labels = ['sed 5m/Myr, 500m basement', 'sed 50m/Myr, 500m basement', 'sed 5m/Myr, 100m basement', 'sed 50m/Myr, 100m basement', 'sed 100m fixed, 500m basement', 'sed 200m fixed, 500m basement', 'sed 50m/Myr, 500m basement, dx fixed at 10km', 'sed 50m/Myr, 500m basement, dx fixed at 20km', 'sed 50m/Myr, 500m basement, dx fixed at 100km', 'sed 100m/Myr, 500m basement, dx fixed at 100km']


txt_labels = ['sed 5m/Myr, 500m basement',
'sed 50m/Myr, 500m basement', 'sed 5m/Myr, 100m basement',
'sed 50m/Myr, 100m basement', 'sed 100m fixed, 500m basement',
'sed 200m fixed, 500m basement',
'sed 100m/Myr, 500m basement, dx fixed at 10km',
'sed 100m/Myr, 500m basement, dx fixed at 25km',
'sed 100m/Myr, 500m basement, dx fixed at 50km',
'sed 100m/Myr, 500m basement, dx fixed at 100km']


# generate qm/ql (hfs_vec)
hfs_vec = np.zeros([len(steps),n_hist])
for i in range(nsteps):
    # n_hist = 0 : fisher2000, sediment 5m/Myr, basement 500m
    hfs_vec[i,0] = 0.43 + 0.0085*age_vec[i]
    #hfs_vec[i,0] = 0.43
    hs_vec[i,0] = 5.0 * age_vec[i]
    dx_vec[i,0] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[0] = 500.0

    # n_hist = 1 : fisher2000, sediment 50m/Myr, basement 500m
    hfs_vec[i,1] = 0.43 + 0.0085*age_vec[i]
    #hfs_vec[i,1] = 0.43
    hs_vec[i,1] = 50.0 * age_vec[i]
    dx_vec[i,1] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[1] = 500.0

    # n_hist = 2 : fisher2000, sediment 5m/Myr, basement 100m
    hfs_vec[i,2] = 0.43 + 0.0085*age_vec[i]
    #hfs_vec[i,2] = 0.43
    hs_vec[i,2] = 5.0 * age_vec[i]
    dx_vec[i,2] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[2] = 100.0

    # n_hist = 3 : fisher2000, sediment 50m/Myr, basement 100m
    hfs_vec[i,3] = 0.43 + 0.0085*age_vec[i]
    #hfs_vec[i,3] = 0.43
    hs_vec[i,3] = 50.0 * age_vec[i]
    dx_vec[i,3] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[3] = 100.0

    # n_hist = 4 : sediment fixed at 100m, basement 500
    hfs_vec[i,4] = 0.43 + 0.0085*age_vec[i]
    #hfs_vec[i,4] = 0.43
    hs_vec[i,4] = 100.0
    dx_vec[i,4] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[4] = 500.0

    # n_hist = 5 : sediment fixed at 200m, basement 500
    hfs_vec[i,5] = 0.43 + 0.0085*age_vec[i]
    #hfs_vec[i,5] = 0.43
    hs_vec[i,5] = 200.0
    dx_vec[i,5] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[5] = 500.0

    # n_hist = 6 : sediment 100m/Myr, dx fixed at 10km
    hfs_vec[i,6] = 0.43 + 0.0085*age_vec[i]
    #hfs_vec[i,6] = 0.43
    hs_vec[i,6] = 100.0 * age_vec[i]
    dx_vec[i,6] = 10000.0 #(5000.0 + 500.0*age_vec[i])/2.0
    param_hb[6] = 500.0

    # n_hist = 7 : sediment 100m/Myr, dx fixed at 25km
    hfs_vec[i,7] = 0.43 + 0.0085*age_vec[i]
    #hfs_vec[i,7] = 0.43
    hs_vec[i,7] = 100.0 * age_vec[i]
    dx_vec[i,7] = 25000.0 #(5000.0 + 500.0*age_vec[i])/2.0
    param_hb[7] = 500.0

    # n_hist = 8 : sediment 100m/Myr, dx fixed at 50km
    hfs_vec[i,8] = 0.43 + 0.0085*age_vec[i]
    #hfs_vec[i,8] = 0.43
    hs_vec[i,8] = 100.0 * age_vec[i]
    dx_vec[i,8] = 50000.0 #(5000.0 + 500.0*age_vec[i])/2.0
    param_hb[8] = 500.0

    # n_hist = 9 : sediment 100m/Myr, dx fixed at 100km
    hfs_vec[i,9] = 0.43 + 0.0085*age_vec[i]
    #hfs_vec[i,9] = 0.43
    hs_vec[i,9] = 100.0 * age_vec[i]
    dx_vec[i,9] = 100000.0 #(5000.0 + 500.0*age_vec[i])/2.0
    param_hb[9] = 500.0





print "age_vec shape" , age_vec.shape
print "hfs_vec shape" , hfs_vec.shape


for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for i in range(len(steps)):
        q_vec[i,j] = (-1.0 * param_lambda * dx_vec[i,j] / (hs_vec[i,j] * param_hb[j] * param_rho * param_c)) * np.log(-hfs_vec[i,j] + 1.0)
        q_vec_log10[i,j] = np.log10(q_vec[i,j])
        q_vec_log10_myr[i,j] = np.log10(q_vec[i,j]*(3.14e7))
    print "done with" , j

# print "q_vec" , q_vec
# print "q_vec_log10 4" , q_vec_log10[:,4]


#hack: FIG: fisher + variations plot
the_lw = 1.75
m_size = 40
hist_begin = 25


fig=plt.figure(figsize=(10.0,4.0))
plt.subplots_adjust(wspace=0.25)

ax1=fig.add_subplot(1, 2, 1, frameon=True)


plt.plot(age_vec[hist_begin:],q_vec_log10[hist_begin:,0], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[hist_begin::100],q_vec_log10[hist_begin::100,0], marker='o', zorder=5, facecolor='k', s=m_size, label=txt_labels[0])

plt.plot(age_vec[hist_begin:],q_vec_log10[hist_begin:,1], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[hist_begin::100],q_vec_log10[hist_begin::100,1], marker='s', zorder=5, facecolor='k', s=m_size, label=txt_labels[1])

plt.plot(age_vec[hist_begin:],q_vec_log10[hist_begin:,2], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[hist_begin::100],q_vec_log10[hist_begin::100,2], marker='o', zorder=5, facecolor='none', s=m_size, label=txt_labels[2])

plt.plot(age_vec[hist_begin:],q_vec_log10[hist_begin:,3], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[hist_begin::100],q_vec_log10[hist_begin::100,3], marker='s', zorder=5, facecolor='none', s=m_size, label=txt_labels[3])

for jj in range(4,10):
    plt.plot(age_vec[hist_begin:],q_vec_log10[hist_begin:,jj], color=plot_col[jj], label=txt_labels[jj], lw=the_lw)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('log10(q [m/s])')
ax1.set_xlabel('age [Myr]')
#plt.legend(fontsize=8,bbox_to_anchor=(1.6, 1.0),ncol=1,columnspacing=0.1)

plt.xlim([np.min(age_vec),np.max(age_vec)])





ax1=fig.add_subplot(1, 2, 2, frameon=True)


plt.plot(age_vec[hist_begin:],q_vec_log10_myr[hist_begin:,0], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[hist_begin::100],q_vec_log10_myr[hist_begin::100,0], marker='o', zorder=5, facecolor='k', s=m_size, label=txt_labels[0])

plt.plot(age_vec[hist_begin:],q_vec_log10_myr[hist_begin:,1], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[hist_begin::100],q_vec_log10_myr[hist_begin::100,1], marker='s', zorder=5, facecolor='k', s=m_size, label=txt_labels[1])

plt.plot(age_vec[hist_begin:],q_vec_log10_myr[hist_begin:,2], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[hist_begin::100],q_vec_log10_myr[hist_begin::100,2], marker='o', zorder=5, facecolor='none', s=m_size, label=txt_labels[2])

plt.plot(age_vec[hist_begin:],q_vec_log10_myr[hist_begin:,3], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[hist_begin::100],q_vec_log10_myr[hist_begin::100,3], marker='s', zorder=5, facecolor='none', s=m_size, label=txt_labels[3])

for jj in range(4,10):
    plt.plot(age_vec[hist_begin:],q_vec_log10_myr[hist_begin:,jj], color=plot_col[jj], label=txt_labels[jj], lw=the_lw)

plt.yticks([-1.0, 0.0, 1.0, 2.0], [0.1, 1.0, 10.0, 100.0])

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('q [m/yr]')
ax1.set_xlabel('age [Myr]')
plt.legend(fontsize=8,bbox_to_anchor=(1.9, 1.0),ncol=1,columnspacing=0.1)

plt.xlim([np.min(age_vec),np.max(age_vec)])



#plt.plot()







# ax1=fig.add_subplot(1, 3, 1, frameon=True)
# ax1.grid()
#
# for jj in [6]:
#     plt.plot(age_vec[50:],q_vec[50:,jj]*(3.14e7), color=plot_col[jj], label=txt_labels[jj], lw=the_lw)
#
# plt.ylabel('q [m/yr]')



plt.savefig(outpath+"q_history.png",bbox_inches='tight')
plt.savefig(outpath+"z_history.eps",bbox_inches='tight')







#todo: path + params
in_path = "../output/revival/winter_basalt_box/"
dir_path = "y_group_d/"

param_strings = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5']
param_nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

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

# print value_dsec[:,:,14,0] - value_dsec_d[:,:,14,0]


#todo: make alt_ind curves
curve_age_vec = np.linspace(0.0,5.0,curve_nsteps)
curve_q_vec = np.zeros([curve_nsteps,n_curves])

# for i in range(curve_nsteps):
for j in range(n_curves):
    f_curve = interpolate.interp1d(curve_age_vec, (q_vec[:,j])*(3.14e7))
    curve_q_vec[:,j] = f_curve(curve_age_vec)

    f_alt_vol = interpolate.interp1d(param_nums,value_alt_vol_mean[:,0,0])
    curve_q_vec_temp = np.zeros(curve_nsteps)
    curve_q_vec_temp[:] = curve_q_vec[:,j]
    curve_q_vec_temp[curve_q_vec_temp > np.max(param_nums)] = np.max(param_nums)
    curve_q_vec_temp[curve_q_vec_temp < np.min(param_nums)] = np.min(param_nums)
    alt_vol_curve[:,j] = f_alt_vol(curve_q_vec_temp)
    #for i in range(curve_nsteps):

    for t in range(len(diff_nums)):
        f_alt_vol = interpolate.interp1d(param_nums,value_alt_vol_mean_d[:,t,0])
        curve_q_vec_temp = np.zeros(curve_nsteps)
        curve_q_vec_temp[:] = curve_q_vec[:,j]
        curve_q_vec_temp[curve_q_vec_temp > np.max(param_nums)] = np.max(param_nums)
        curve_q_vec_temp[curve_q_vec_temp < np.min(param_nums)] = np.min(param_nums)
        alt_vol_curve_diff_d[:,j,t] = f_alt_vol(curve_q_vec_temp)


#todo: make dsec_curves
for jj in range(len(any_min)):

    for j in range(n_curves):
        f_curve = interpolate.interp1d(curve_age_vec, (q_vec[:,j])*(3.14e7))
        curve_q_vec[:,j] = f_curve(curve_age_vec)

        f_dsec = interpolate.interp1d(param_nums,value_dsec[:,0,any_min[jj],0])
        curve_q_vec_temp = np.zeros(curve_nsteps)
        curve_q_vec_temp[:] = curve_q_vec[:,j]
        curve_q_vec_temp[curve_q_vec_temp > np.max(param_nums)] = np.max(param_nums)
        curve_q_vec_temp[curve_q_vec_temp < np.min(param_nums)] = np.min(param_nums)
        dsec_curve[:,j,any_min[jj]] = f_dsec(curve_q_vec_temp)
        #for i in range(curve_nsteps):

        # for t in range(len(diff_nums)):
        #     f_alt_vol = interpolate.interp1d(param_nums,value_alt_vol_mean_d[:,t,0])
        #     curve_q_vec_temp = np.zeros(curve_nsteps)
        #     curve_q_vec_temp[:] = curve_q_vec[:,j]
        #     curve_q_vec_temp[curve_q_vec_temp > np.max(param_nums)] = np.max(param_nums)
        #     curve_q_vec_temp[curve_q_vec_temp < np.min(param_nums)] = np.min(param_nums)
        #     alt_vol_curve_diff_d[:,j,t] = f_alt_vol(curve_q_vec_temp)

# print "dsec_curve[:,6,1]"
# print dsec_curve[:,6,1]
# print " "




#hack: FIG: curves

m_skip = 50
m_size = 30


fig=plt.figure(figsize=(10.0,10.0))

ax=fig.add_subplot(2, 2, 1, frameon=True)
plt.grid()



plt.plot(curve_age_vec[100:],curve_q_vec[100:,0], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],curve_q_vec[100::m_skip,0], marker='o', zorder=5, facecolor='k', s=m_size, label=txt_labels[0])

plt.plot(curve_age_vec[100:],curve_q_vec[100:,1], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],curve_q_vec[100::m_skip,1], marker='s', zorder=5, facecolor='k', s=m_size, label=txt_labels[1])

plt.plot(curve_age_vec[100:],curve_q_vec[100:,2], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],curve_q_vec[100::m_skip,2], marker='o', zorder=5, facecolor='none', s=m_size, label=txt_labels[2])

plt.plot(curve_age_vec[100:],curve_q_vec[100:,3], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],curve_q_vec[100::m_skip,3], marker='s', zorder=5, facecolor='none', s=m_size, label=txt_labels[3])

for j in range(4,10):
    plt.plot(curve_age_vec[100:],curve_q_vec[100:,j], color=plot_col[j], label=txt_labels[j], lw=2.0)

plt.xlim([0.5,5.0])
plt.ylim([0.0,10.0])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('age [Myr]')
plt.ylabel('q [m/yr]')



ax=fig.add_subplot(2, 2, 2, frameon=True)
plt.grid()
# for j in range(n_curves):
#     plt.plot(curve_age_vec[100:],alt_vol_curve[100:,j], color=plot_col[j], label=txt_labels[j], lw=2.0)

plt.plot(curve_age_vec[100:],alt_vol_curve[100:,0], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],alt_vol_curve[100::m_skip,0], marker='o', zorder=5, facecolor='k', s=m_size, label=txt_labels[0])

plt.plot(curve_age_vec[100:],alt_vol_curve[100:,1], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],alt_vol_curve[100::m_skip,1], marker='s', zorder=5, facecolor='k', s=m_size, label=txt_labels[1])

plt.plot(curve_age_vec[100:],alt_vol_curve[100:,2], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],alt_vol_curve[100::m_skip,2], marker='o', zorder=5, facecolor='none', s=m_size, label=txt_labels[2])

plt.plot(curve_age_vec[100:],alt_vol_curve[100:,3], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],alt_vol_curve[100::m_skip,3], marker='s', zorder=5, facecolor='none', s=m_size, label=txt_labels[3])

for j in range(4,10):
    plt.plot(curve_age_vec[100:],alt_vol_curve[100:,j], color=plot_col[j], label=txt_labels[j], lw=2.0)

plt.xlim([0.5,5.0])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('age [Myr]')
plt.ylabel('alt_vol slope')
plt.legend(fontsize=9,bbox_to_anchor=(2.0, 1.0),ncol=1,columnspacing=0.1)






#poop: plot dsec curves TEST
ax=fig.add_subplot(2, 2, 4, frameon=True)
plt.grid()
# for j in range(n_curves):
#     plt.plot(curve_age_vec[100:],alt_vol_curve[100:,j], color=plot_col[j], label=txt_labels[j], lw=2.0)

this_min = any_min[0]
print "this_min curve test" , this_min

plt.plot(curve_age_vec[100:],dsec_curve[100:,0,this_min], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],dsec_curve[100::m_skip,0,this_min], marker='o', zorder=5, facecolor='k', s=m_size, label=txt_labels[0])

plt.plot(curve_age_vec[100:],dsec_curve[100:,1,this_min], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],dsec_curve[100::m_skip,1,this_min], marker='s', zorder=5, facecolor='k', s=m_size, label=txt_labels[1])

plt.plot(curve_age_vec[100:],dsec_curve[100:,2,this_min], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],dsec_curve[100::m_skip,2,this_min], marker='o', zorder=5, facecolor='none', s=m_size, label=txt_labels[2])

plt.plot(curve_age_vec[100:],dsec_curve[100:,3,this_min], color=plot_col[0], lw=the_lw)
plt.scatter(curve_age_vec[100::m_skip],dsec_curve[100::m_skip,3,this_min], marker='s', zorder=5, facecolor='none', s=m_size, label=txt_labels[3])

# print "dsec_curve[100:,6,this_min]"
# print dsec_curve[100:,6,this_min]
# print " "


for j in range(4,10):
#for j in [6]:
    plt.plot(curve_age_vec[100:],dsec_curve[100:,j,this_min], color=plot_col[j], label=txt_labels[j], lw=2.0)

plt.xlim([0.5,5.0])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('age [Myr]')
plt.ylabel('test rate ' + secondary[this_min])
# plt.legend(fontsize=9,bbox_to_anchor=(2.0, 1.0),ncol=1,columnspacing=0.1)









#poop: colormap
# cmap1 = LinearSegmentedColormap.from_list("my_colormap", ((28.0/255.0, 207.0/255.0, 94.0/255.0), (26.0/255.0, 179.0/255.0, 189.0/255.0), (63.0/255.0, 35.0/255.0, 108.0/255.0)), N=6, gamma=1.0)
cmap1 = LinearSegmentedColormap.from_list("my_colormap", ((0.64, 0.1, 0.53), (0.78, 0.61, 0.02)), N=15, gamma=1.0)
diff_colors = [ cmap1(x) for x in np.linspace(0.0, 1.0, cont_x_diff_max) ]

ax=fig.add_subplot(2, 2, 3, frameon=True)
plt.grid()
for t in range(cont_x_diff_max):
    plt.plot(curve_age_vec[100:],alt_vol_curve_diff_d[100:,6,t], color=diff_colors[t], label="mix time = 10**" + diff_strings[t] + " [years]", lw=2.0)
# plt.legend(fontsize=8,bbox_to_anchor=(1.8, 1.0),ncol=1,columnspacing=0.1)
plt.legend(fontsize=9,loc='best',ncol=1,columnspacing=0.1)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('age [Myr]')
plt.ylabel('alt_vol slope')
plt.title(txt_labels[6], color=plot_col[6], fontsize=11)

# print value_alt_vol_mean[:,0,0]
# print param_nums

plt.savefig(outpath+"q_curves.png",bbox_inches='tight')
plt.savefig(outpath+"z_curves.eps",bbox_inches='tight')










#hack: FIG: dsec_curves

m_skip = 100
m_size = 25
the_lw = 1.2

fig=plt.figure(figsize=(12.0,12.0))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

sp_a = (len(any_min)+3.0)/4.0
sp_b = 4

for jj in range(len(any_min)):

    ax=fig.add_subplot(sp_a, sp_b, jj+1, frameon=True)
    plt.grid()

    this_min = any_min[jj]
    print "this_min curve real" , this_min

    plt.plot(curve_age_vec[100:],dsec_curve[100:,0,this_min], color=plot_col[0], lw=the_lw)
    plt.scatter(curve_age_vec[100::m_skip],dsec_curve[100::m_skip,0,this_min], marker='o', zorder=5, facecolor='k', s=m_size, label=txt_labels[0])

    plt.plot(curve_age_vec[100:],dsec_curve[100:,1,this_min], color=plot_col[0], lw=the_lw)
    plt.scatter(curve_age_vec[100::m_skip],dsec_curve[100::m_skip,1,this_min], marker='s', zorder=5, facecolor='k', s=m_size, label=txt_labels[1])

    plt.plot(curve_age_vec[100:],dsec_curve[100:,2,this_min], color=plot_col[0], lw=the_lw)
    plt.scatter(curve_age_vec[100::m_skip],dsec_curve[100::m_skip,2,this_min], marker='o', zorder=5, facecolor='none', s=m_size, label=txt_labels[2])

    plt.plot(curve_age_vec[100:],dsec_curve[100:,3,this_min], color=plot_col[0], lw=the_lw)
    plt.scatter(curve_age_vec[100::m_skip],dsec_curve[100::m_skip,3,this_min], marker='s', zorder=5, facecolor='none', s=m_size, label=txt_labels[3])

    for j in range(4,10):
    #for j in [6]:
        plt.plot(curve_age_vec[100:],dsec_curve[100:,j,this_min], color=plot_col[j], label=txt_labels[j], lw=the_lw)

    plt.xlim([0.5,5.0])
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel('age [Myr]',fontsize=7)
    # plt.ylabel('rate',fontsize=7)
    plt.title(secondary[any_min[jj]])



plt.savefig(outpath+"q_dsec_curves.png",bbox_inches='tight')
plt.savefig(outpath+"z_dsec_curves.eps",bbox_inches='tight')










#todo: make curve arrays etc.

n_curves = 10
n_curves_x = 3

y_param_paths = np.zeros([curve_nsteps,n_curves,n_curves_x])
x_diff_paths = np.zeros([curve_nsteps,n_curves,n_curves_x])

y_param_paths[:,6,0] = curve_q_vec[:,6]
y_param_paths[:,7,0] = curve_q_vec[:,7]
y_param_paths[:,8,0] = curve_q_vec[:,8]
y_param_paths[:,9,0] = curve_q_vec[:,9]

y_param_paths[:,6,1] = curve_q_vec[:,6]
y_param_paths[:,7,1] = curve_q_vec[:,7]
y_param_paths[:,8,1] = curve_q_vec[:,8]
y_param_paths[:,9,1] = curve_q_vec[:,9]

y_param_paths[:,6,2] = curve_q_vec[:,6]
y_param_paths[:,7,2] = curve_q_vec[:,7]
y_param_paths[:,8,2] = curve_q_vec[:,8]
y_param_paths[:,9,2] = curve_q_vec[:,9]

# y_param_paths[:,16] = curve_q_vec[:,6]
# y_param_paths[:,17] = curve_q_vec[:,7]
# y_param_paths[:,18] = curve_q_vec[:,8]
# y_param_paths[:,19] = curve_q_vec[:,9]
#
# y_param_paths[:,26] = curve_q_vec[:,6]
# y_param_paths[:,27] = curve_q_vec[:,7]
# y_param_paths[:,28] = curve_q_vec[:,8]
# y_param_paths[:,29] = curve_q_vec[:,9]

curves_saved = np.zeros([curve_nsteps,n_curves,n_curves_x])
curves_saved_d = np.zeros([curve_nsteps,n_curves,n_curves_x])
curves_saved_a = np.zeros([curve_nsteps,n_curves,n_curves_x])
curves_saved_b = np.zeros([curve_nsteps,n_curves,n_curves_x])

curves_total_saved = np.zeros([curve_nsteps,n_curves,n_curves_x])
curves_total_saved_d = np.zeros([curve_nsteps,n_curves,n_curves_x])
curves_total_saved_a = np.zeros([curve_nsteps,n_curves,n_curves_x])
curves_total_saved_b = np.zeros([curve_nsteps,n_curves,n_curves_x])

curves_dsec_saved = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])
curves_dsec_saved_d = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])
curves_dsec_saved_a = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])
curves_dsec_saved_b = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])

curves_dsec_n_saved = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])
curves_dsec_n_saved_d = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])
curves_dsec_n_saved_a = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])
curves_dsec_n_saved_b = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])

curves_dsec_bin_saved = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])
curves_dsec_bin_saved_d = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])
curves_dsec_bin_saved_a = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])
curves_dsec_bin_saved_b = np.zeros([curve_nsteps,minNum+1,n_curves,n_curves_x])


interp_start = 100

test_grid = np.zeros([900,900,10])
test_grid_d = np.zeros([900,900,10])

#todo: fill curves_saved
interp_kind_alt = 'linear'
for c in [6, 7, 8, 9]:
    for cc in [0, 1, 2]:
        y_param_path_temp = y_param_paths[:,c,cc]


        if cc == 0:
            x_diff_path_temp_d = np.linspace(2.0,5.0,num=curve_nsteps,endpoint=True)#3.5*np.ones(curve_nsteps)
        if cc == 1:
            x_diff_path_temp_d = 2.5*np.ones(curve_nsteps)
        if cc == 2:
            x_diff_path_temp_d = 4.0*np.ones(curve_nsteps)

        x_diff_path_temp = 1.0*np.ones(curve_nsteps)
        curve1 = any_2d_interp(diff_nums, param_nums, value_alt_vol_mean[:,:,0], x_diff_path_temp[interp_start:], y_param_path_temp[interp_start:],kind_in=interp_kind_alt)
        curve1_vector = curve1[::-1,0]
        curves_saved[interp_start:,c,cc] = curve1_vector




        curve2 = any_2d_interp(diff_nums, param_nums, value_alt_vol_mean_d[:,:,0], x_diff_path_temp_d[interp_start:], y_param_path_temp[interp_start:],kind_in=interp_kind_alt)
        # curve2_vector = curve2[::-1,0]
        xxx_grid, yyy_grid = np.meshgrid(diff_nums,param_nums)
        # print "xxx_grid.shape" , xxx_grid.shape
        # print "yyy_grid.shape" , yyy_grid.shape
        # print "value_dsec_d[:,:,any_min[j],0].shape" , value_dsec_d[:,:,any_min[j],0].shape
        curve2_interp = interpolate.interp2d(diff_nums, param_nums, value_alt_vol_mean_d[:,:,0], kind=interp_kind_alt)
        curve2_vector = np.zeros(900)
        for iii in range(900):
            x_interp_temp = x_diff_path_temp_d[interp_start+iii]
            y_interp_temp = y_param_path_temp[interp_start+iii]
            curve2_vector[iii] = curve2_interp(x_interp_temp,y_interp_temp)
            if x_interp_temp >= np.max(diff_nums):
                x_interp_temp = np.max(diff_nums)
                curve2_vector[iii] = curve2_vector[iii-1]

            if x_interp_temp <= np.min(diff_nums):
                x_interp_temp = np.min(diff_nums)
                curve2_vector[iii] = curve2_vector[iii-1]

            if y_interp_temp >= np.max(param_nums):
                y_interp_temp = np.max(param_nums)
                curve2_vector[iii] = curve2_vector[iii-1]

            if y_interp_temp <= np.min(param_nums):
                y_interp_temp = np.min(param_nums)
                curve2_vector[iii] = curve2_vector[iii-1]



        for iii in range(900-1):
            if curve2_vector[-iii] == 0.0:
                curve2_vector[-iii] = curve2_vector[-iii+1]
        curves_saved_d[interp_start:,c,cc] = curve2_vector


        #todo: fill curves_dsec_saved

        for j in range(len(any_min)):
            curve1 = any_2d_interp(diff_nums, param_nums, value_dsec[:,:,any_min[j],0], x_diff_path_temp[interp_start:], y_param_path_temp[interp_start:],kind_in=interp_kind_alt)

            # if any_min[j] == 11:
            #     test_grid[:,:,c,cc] = curve1

            curve1_vector = curve1[::-1,0]
            curves_dsec_saved[interp_start:,any_min[j],c,cc] = curve1_vector

            curve2 = any_2d_interp(diff_nums, param_nums, value_dsec_d[:,:,any_min[j],0], x_diff_path_temp_d[interp_start:], y_param_path_temp[interp_start:],kind_in=interp_kind_alt)
            xxx_grid, yyy_grid = np.meshgrid(diff_nums,param_nums)
            # print "xxx_grid.shape" , xxx_grid.shape
            # print "yyy_grid.shape" , yyy_grid.shape
            # print "value_dsec_d[:,:,any_min[j],0].shape" , value_dsec_d[:,:,any_min[j],0].shape
            curve2_interp = interpolate.interp2d(diff_nums, param_nums, value_dsec_d[:,:,any_min[j],0], kind=interp_kind_alt)
            curve2_vector = np.zeros(900)
            for iii in range(900):
                x_interp_temp = x_diff_path_temp_d[interp_start+iii]
                y_interp_temp = y_param_path_temp[interp_start+iii]
                curve2_vector[iii] = curve2_interp(x_interp_temp,y_interp_temp)
                if x_interp_temp >= np.max(diff_nums):
                    x_interp_temp = np.max(diff_nums)
                    curve2_vector[iii] = curve2_vector[iii-1]

                if x_interp_temp <= np.min(diff_nums):
                    x_interp_temp = np.min(diff_nums)
                    curve2_vector[iii] = curve2_vector[iii-1]

                if y_interp_temp >= np.max(param_nums):
                    y_interp_temp = np.max(param_nums)
                    curve2_vector[iii] = curve2_vector[iii-1]

                if y_interp_temp <= np.min(param_nums):
                    y_interp_temp = np.min(param_nums)
                    curve2_vector[iii] = curve2_vector[iii-1]

            for iii in range(900-1):
                if curve2_vector[-iii] == 0.0:
                    curve2_vector[-iii] = curve2_vector[-iii+1]
                # x_interp_temp = 3.0
                # y_interp_temp = 3.0

                # curve2_vector[iii] = curve2[iii,-iii]
            # curve2_vector = curve2_vector[::-1]
            curves_dsec_saved_d[interp_start:,any_min[j],c,cc] = curve2_vector

            # if any_min[j] == 17:
            #     test_grid_d[:,:,c,cc] = curve2


            max_temp = np.max(curves_dsec_saved[interp_start:,any_min[j],c,cc])
            if np.max(curves_dsec_saved_d[interp_start:,any_min[j],c,cc]) > max_temp:
                max_temp = np.max(curves_dsec_saved_d[interp_start:,any_min[j],c,cc])
            for ii in range(interp_start,curve_nsteps):
                #poop: normalize dsec curves here

                # if curves_dsec_saved[ii,any_min[j],c,cc] > 0.0:
                #     curves_dsec_n_saved[ii,any_min[j],c,cc] = curves_dsec_saved[ii,any_min[j],c,cc]/max_temp
                curves_dsec_n_saved[ii,any_min[j],c,cc] = curves_dsec_saved[ii,any_min[j],c,cc]

            for ii in range(interp_start,curve_nsteps):
                # if curves_dsec_saved_d[ii,any_min[j],c,cc] > 0.0:
                #     curves_dsec_n_saved_d[ii,any_min[j],c,cc] = curves_dsec_saved_d[ii,any_min[j],c,cc]/max_temp
                curves_dsec_n_saved_d[ii,any_min[j],c,cc] = curves_dsec_saved_d[ii,any_min[j],c,cc]



# fig=plt.figure(figsize=(16.0,4.0))
# test_skip = 4
#
# ax=fig.add_subplot(1, 4, 1, frameon=True)
# plt.pcolor(test_grid[::test_skip,::test_skip,6])
# plt.colorbar(orientation='vertical')
# plt.title('6')
#
# ax=fig.add_subplot(1, 4, 2, frameon=True)
# plt.pcolor(test_grid[::test_skip,::test_skip,7])
# plt.colorbar(orientation='vertical')
# plt.title('7')
#
# ax=fig.add_subplot(1, 4, 3, frameon=True)
# plt.pcolor(test_grid[::test_skip,::test_skip,8])
# plt.colorbar(orientation='vertical')
# plt.title('8')
#
# ax=fig.add_subplot(1, 4, 4, frameon=True)
# plt.pcolor(test_grid[::test_skip,::test_skip,9])
# plt.colorbar(orientation='vertical')
# plt.title('9')
# plt.savefig(outpath+"p_test_grid.png",bbox_inches='tight')
#
#
#
# fig=plt.figure(figsize=(16.0,4.0))
# test_skip = 4
#
# ax=fig.add_subplot(1, 4, 1, frameon=True)
# plt.pcolor(test_grid_d[::test_skip,::test_skip,6])
# plt.colorbar(orientation='vertical')
# plt.title('6')
#
# ax=fig.add_subplot(1, 4, 2, frameon=True)
# plt.pcolor(test_grid_d[::test_skip,::test_skip,7])
# plt.colorbar(orientation='vertical')
# plt.title('7')
#
# ax=fig.add_subplot(1, 4, 3, frameon=True)
# plt.pcolor(test_grid_d[::test_skip,::test_skip,8])
# plt.colorbar(orientation='vertical')
# plt.title('8')
#
# ax=fig.add_subplot(1, 4, 4, frameon=True)
# plt.pcolor(test_grid_d[::test_skip,::test_skip,9])
# plt.colorbar(orientation='vertical')
# plt.title('9')
# plt.savefig(outpath+"p_test_grid_d.png",bbox_inches='tight')


cc = 0


#hack: FIG: q_interp2d_curves
path_skip = 100

# cmap2 = LinearSegmentedColormap.from_list("my_colormap", ((0.8, 0.0, 0.0), (0.88, 0.59, 0.15)), N=15, gamma=1.0)
cmap2 = LinearSegmentedColormap.from_list("my_colormap", ((0.59, 0.02, 0.11), (1.0, 0.75, 0.0)), N=15, gamma=1.0)
age_colors = [ cmap2(x) for x in np.linspace(0.0, 1.0, 10) ]
age_size = 45

fig=plt.figure(figsize=(12.0,8))
#plt.subplots_adjust(wspace=0.3, hspace=0.3)


ax=fig.add_subplot(2, 3, 1, frameon=True)
plt.grid()

for c in [6, 7, 8, 9]:
# for c in [7]:
    plt.plot(age_vec[interp_start:],curves_saved[interp_start:,c,cc], color=plot_col[c],label=txt_labels[c], lw=2)
    plt.scatter(age_vec[interp_start::path_skip], curves_saved[interp_start::path_skip,c,cc], marker='o', s=age_size, facecolors=age_colors, edgecolor='none', zorder=5)

plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Alteration Rate', fontsize=9)
plt.title('interpolated alteration rate histories, solo')
plt.xlim([0.5,5.0])
plt.ylim([5.0,10.0])


ax=fig.add_subplot(2, 3, 2, frameon=True)
plt.grid()

for c in [6, 7, 8, 9]:
    plt.plot(age_vec[interp_start:],curves_saved_d[interp_start:,c,cc], color=plot_col[c],label=txt_labels[c], lw=2)
    plt.scatter(age_vec[interp_start::path_skip], curves_saved_d[interp_start::path_skip,c,cc], marker='o', s=age_size, facecolors=age_colors, edgecolor='none', zorder=5)

plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Alteration Rate', fontsize=9)
plt.title('interpolated alteration rate histories, dual')
plt.xlim([0.5,5.0])
plt.ylim([5.0,10.0])

# COMBINED COLUMN HERE?
ax=fig.add_subplot(2, 3, 3, frameon=True)
plt.grid()

for c in [6, 7, 8, 9]:
    plt.plot(age_vec[interp_start:],curves_saved[interp_start:,c,cc], color=plot_col[c],label=txt_labels[c], lw=2)
    plt.plot(age_vec[interp_start:],curves_saved_d[interp_start:,c,cc], color=plot_col[c],label=txt_labels[c], lw=2, linestyle='--')

plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Alteration Rate', fontsize=9)
plt.title('interpolated alteration rate histories, dual')
plt.xlim([0.5,5.0])
plt.ylim([5.0,10.0])
# plt.legend(fontsize=9,bbox_to_anchor=(1.0, -0.1),ncol=1,columnspacing=0.0,labelspacing=0.0)










# PLOT ALL REGULAR AGE PATHS IN THE VERTICAL WITH ANNOTATIONS
ax=fig.add_subplot(2, 3, 4, frameon=True)


c_count = 2.0
for c in [6, 7, 8, 9]:
    c_count = c_count + 0.5
    plt.plot(c_count*x_diff_path_temp, y_param_paths[:,c,cc], color=plot_col[c], label=txt_labels[c], lw=2)
    # plt.scatter(scatter_x[::path_skip], y_param_paths[::path_skip,c], marker='o', s=40, facecolor=plot_col[c], edgecolor='none')
    plt.scatter(c_count*x_diff_path_temp[::path_skip], y_param_paths[::path_skip,c,cc], marker='o', s=age_size, facecolors=age_colors, edgecolor='none', zorder=5)


    age_marker_labels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for i, txt in enumerate(age_marker_labels):
        ii = (i+1)*100 - 1
        ax.annotate(txt, (c_count*x_diff_path_temp[ii],y_param_paths[ii,c,cc]), xytext=(5, 0), textcoords='offset points',fontsize=8)

plt.legend(fontsize=9,bbox_to_anchor=(1.0, -0.1),ncol=1,columnspacing=0.0,labelspacing=0.0)


plt.xlim([np.min(diff_nums), np.max(diff_nums)])
plt.ylim([np.min(param_nums), np.max(param_nums)])

plt.xlabel('log10(mixing time [years])', fontsize=9)
plt.ylabel('discharge q [m/yr]', fontsize=9)
plt.title('paths through solo parameter space')













# PLOT ALL REGULAR AGE PATHS IN THE VERTICAL WITH ANNOTATIONS
ax=fig.add_subplot(2, 3, 5, frameon=True)

c_count = 2.0
for c in [6, 7, 8, 9]:
    c_count = c_count + 0.5
    plt.plot(x_diff_path_temp_d, y_param_paths[:,c,cc], color=plot_col[c], label=txt_labels[c], lw=2)
    plt.scatter(x_diff_path_temp_d[::path_skip], y_param_paths[::path_skip,c,cc], marker='o', s=age_size, facecolors=age_colors, edgecolor='none', zorder=5)


    age_marker_labels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for i, txt in enumerate(age_marker_labels):
        ii = (i+1)*100 - 1
        ax.annotate(txt, (x_diff_path_temp_d[ii],y_param_paths[ii,c,cc]), xytext=(5, 0), textcoords='offset points',fontsize=8)

plt.legend(fontsize=9,bbox_to_anchor=(1.0, -0.1),ncol=1,columnspacing=0.0,labelspacing=0.0)


plt.xlim([np.min(diff_nums), np.max(diff_nums)])
plt.ylim([np.min(param_nums), np.max(param_nums)])

plt.xlabel('log10(mixing time [years])', fontsize=9)
plt.ylabel('discharge q [m/yr]', fontsize=9)
plt.title('paths through dual parameter space')



plt.savefig(outpath+"q_interp2d_curves.png",bbox_inches='tight')
plt.savefig(outpath+"z_interp2d_curves.eps",bbox_inches='tight')






#hack: 2D dsec curve figure!

fig=plt.figure(figsize=(16.0,12.0*2.0/3.0))

c_plot = 1
for c in [6, 7, 8, 9]:
    ax=fig.add_subplot(3, 4, c_plot, frameon=True)
    c_plot = c_plot + 1
    plt.grid()
    for j in range(len(any_min)):
        plt.plot(age_vec[interp_start:],curves_dsec_n_saved[interp_start:,any_min[j],c,cc], color=col[j],label=secondary[any_min[j]], lw=1.5)
        #plt.scatter(age_vec[interp_start::path_skip], curves_dsec_saved[interp_start::path_skip,any_min[j],c], marker='o', s=40, facecolor=plot_col[c], edgecolor='none')

    # plt.xlabel('Age [Myr]', fontsize=9)
    plt.ylabel('Secondary Growth Rate', fontsize=9)
    plt.title('interpolated growth rate histories solo', color=plot_col[c])
    plt.xlim([0.5,5.0])
    if c == 6:
        plt.legend(fontsize=8,bbox_to_anchor=(1.3, 1.4),ncol=4,columnspacing=0.0,labelspacing=0.0)


c_plot = 1
for c in [6, 7, 8, 9]:
    ax=fig.add_subplot(3, 4, 4+c_plot, frameon=True)
    c_plot = c_plot + 1
    plt.grid()
    for j in range(len(any_min)):
        plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,any_min[j],c,cc], color=col[j],label=secondary[any_min[j]], lw=1.5)
        #plt.scatter(age_vec[interp_start::path_skip], curves_dsec_saved[interp_start::path_skip,any_min[j],c], marker='o', s=40, facecolor=plot_col[c], edgecolor='none')

    # plt.xlabel('Age [Myr]', fontsize=9)
    plt.ylabel('Secondary Growth Rate', fontsize=9)
    plt.title('interpolated growth rate histories dual', color=plot_col[c])
    plt.xlim([0.5,5.0])
    # if c == 6:
    #     plt.legend(fontsize=8,bbox_to_anchor=(1.0, 1.4),ncol=4,columnspacing=0.0,labelspacing=0.0)


c_plot = 1
for c in [6, 7, 8, 9]:
    ax=fig.add_subplot(3, 4, 8+c_plot, frameon=True)
    c_plot = c_plot + 1
    #plt.grid()
    for j in range(len(any_min)):
        plt.plot(age_vec[interp_start:],curves_dsec_n_saved[interp_start:,any_min[j],c,cc], color=col[j],label=secondary[any_min[j]], lw=1.5)

        plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,any_min[j],c,cc], color=col[j],label=secondary[any_min[j]], lw=1.5,linestyle='--')
        #plt.scatter(age_vec[interp_start::path_skip], curves_dsec_saved[interp_start::path_skip,any_min[j],c], marker='o', s=40, facecolor=plot_col[c], edgecolor='none')

    plt.xlabel('Age [Myr]', fontsize=9)
    plt.ylabel('Secondary Growth Rate', fontsize=9)
    plt.title('interpolated growth rate histories dual', color=plot_col[c])
    plt.xlim([0.5,5.0])



plt.savefig(outpath+"q_interp2d_dsec.png",bbox_inches='tight')
plt.savefig(outpath+"z_interp2d_dsec.eps",bbox_inches='tight')









#hack: 2D dsec curve ALT!

fig=plt.figure(figsize=(16.0,12.0*2.0/3.0))
plt.subplots_adjust(hspace=0.4)

the_dashes = [6, 2]



ax=fig.add_subplot(3, 4, 1, frameon=True)
#plt.grid()
cc = 0
for c in [6, 7, 8, 9]:
    # for cc in [1, 2]:
    if c == 6:
        plot_hem = plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,14,c,cc], color=plot_col[c],label='FC dual 6-9', lw=1.2, dashes=the_dashes)
    if c > 6:
        plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,14,c,cc], color=plot_col[c], lw=1.2,  dashes=the_dashes)
plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Secondary Growth Rate', fontsize=9)
plt.title('Fe-Celadonite main paths', fontsize=10)
plt.xlim([0.5,5.0])
#plt.ylim([0.0,1.0])
plt.legend( fontsize=10,bbox_to_anchor=(1.0, 1.4),ncol=1,columnspacing=0.0,labelspacing=0.0)



ax=fig.add_subplot(3, 4, 2, frameon=True)
#plt.grid()
cc = 1
for c in [6, 7, 8, 9]:
    # for cc in [1, 2]:
    if c == 6:
        plot_hem = plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,14,c,cc], color=plot_col[c],label='FC dual 6-9', lw=1.2, dashes=the_dashes)
    if c > 6:
        plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,14,c,cc], color=plot_col[c], lw=1.2,  dashes=the_dashes)
plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Secondary Growth Rate', fontsize=9)
plt.title('Fe-Celadonite vertical left', fontsize=10)
plt.xlim([0.5,5.0])
#plt.ylim([0.0,1.0])



ax=fig.add_subplot(3, 4, 3, frameon=True)
#plt.grid()
cc = 2
for c in [6, 7, 8, 9]:
    # for cc in [1, 2]:
    if c == 6:
        plot_hem = plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,14,c,cc], color=plot_col[c],label='FC dual 6-9', lw=1.2, dashes=the_dashes)
    if c > 6:
        plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,14,c,cc], color=plot_col[c], lw=1.2,  dashes=the_dashes)
plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Secondary Growth Rate', fontsize=9)
plt.title('Fe-Celadonite vertical right', fontsize=10)
plt.xlim([0.5,5.0])
#plt.ylim([0.0,1.0])




ax=fig.add_subplot(3, 4, 4, frameon=True)
#plt.grid()
cc = 1
for c in [6, 7, 8, 9]:
    plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,14,c,cc], color=plot_col[c], lw=1.2)
cc = 2
for c in [6, 7, 8, 9]:
    plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,14,c,cc], color=plot_col[c], lw=1.2,  dashes=the_dashes)
plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Secondary Growth Rate', fontsize=9)
plt.title('FC left (solid) + right (dashed)', fontsize=10)
plt.xlim([0.5,5.0])
#plt.ylim([0.0,1.0])







ax=fig.add_subplot(3, 4, 5, frameon=True)
#plt.grid()
cc = 0
for c in [6, 7, 8, 9]:
    # for cc in [1, 2]:
    if c == 6:
        plot_hem = plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,17,c,cc], color=plot_col[c],label='hem dual 6-9', lw=1.2, dashes=the_dashes)
    if c > 6:
        plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,14,c,cc], color=plot_col[c], lw=1.2,  dashes=the_dashes)
plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Secondary Growth Rate', fontsize=9)
plt.title('Hem main paths', fontsize=10)
plt.xlim([0.5,5.0])
#plt.ylim([0.0,1.0])
plt.legend( fontsize=10,bbox_to_anchor=(1.0, 1.4),ncol=1,columnspacing=0.0,labelspacing=0.0)



ax=fig.add_subplot(3, 4, 6, frameon=True)
#plt.grid()
cc = 1
for c in [6, 7, 8, 9]:
    # for cc in [1, 2]:
    if c == 6:
        plot_hem = plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,17,c,cc], color=plot_col[c],label='hem dual 6-9', lw=1.2, dashes=the_dashes)
    if c > 6:
        plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,14,c,cc], color=plot_col[c], lw=1.2,  dashes=the_dashes)
plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Secondary Growth Rate', fontsize=9)
plt.title('Hem vertical left', fontsize=10)
plt.xlim([0.5,5.0])
#plt.ylim([0.0,1.0])



ax=fig.add_subplot(3, 4, 7, frameon=True)
#plt.grid()
cc = 2
for c in [6, 7, 8, 9]:
    # for cc in [1, 2]:
    if c == 6:
        plot_hem = plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,17,c,cc], color=plot_col[c],label='hem dual 6-9', lw=1.2, dashes=the_dashes)
    if c > 6:
        plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,14,c,cc], color=plot_col[c], lw=1.2,  dashes=the_dashes)
plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Secondary Growth Rate', fontsize=9)
plt.title('Hem vertical right', fontsize=10)
plt.xlim([0.5,5.0])
#plt.ylim([0.0,1.0])



ax=fig.add_subplot(3, 4, 8, frameon=True)
#plt.grid()
cc = 1
for c in [6, 7, 8, 9]:
    plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,17,c,cc], color=plot_col[c], lw=1.2)
cc = 2
for c in [6, 7, 8, 9]:
    plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,17,c,cc], color=plot_col[c], lw=1.2,  dashes=the_dashes)
plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Secondary Growth Rate', fontsize=9)
plt.title('hem left (solid) + right (dashed)', fontsize=10)
plt.xlim([0.5,5.0])
#plt.ylim([0.0,1.0])


#plt.legend( fontsize=10,bbox_to_anchor=(1.0, 1.4),ncol=1,columnspacing=0.0,labelspacing=0.0)

#plt.legend( fontsize=10,bbox_to_anchor=(1.0, 1.4),ncol=1,columnspacing=0.0,labelspacing=0.0)

# c_plot = 1
# for c in [6, 7, 8, 9]:
#     ax=fig.add_subplot(3, 4, 2, frameon=True)
#     plt.grid()
#     for j in range(len(any_min)):
#         plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,any_min[j],c], color=col[j],label=secondary[any_min[j]], lw=1.5)
#         #plt.scatter(age_vec[interp_start::path_skip], curves_dsec_saved[interp_start::path_skip,any_min[j],c], marker='o', s=40, facecolor=plot_col[c], edgecolor='none')
#
#     # plt.xlabel('Age [Myr]', fontsize=9)
#     plt.ylabel('Secondary Growth Rate', fontsize=9)
#     plt.title('interpolated growth rate histories dual', color=plot_col[c])
#     plt.xlim([0.5,5.0])
#     # if c == 6:
#     #     plt.legend(fontsize=8,bbox_to_anchor=(1.0, 1.4),ncol=4,columnspacing=0.0,labelspacing=0.0)
#
#
# c_plot = 1
# for c in [6, 7, 8, 9]:
#     ax=fig.add_subplot(3, 4, 3, frameon=True)
#     #plt.grid()
#     for j in range(len(any_min)):
#         plt.plot(age_vec[interp_start:],curves_dsec_n_saved[interp_start:,any_min[j],c], color=col[j],label=secondary[any_min[j]], lw=1.5)
#
#         plt.plot(age_vec[interp_start:],curves_dsec_n_saved_d[interp_start:,any_min[j],c], color=col[j],label=secondary[any_min[j]], lw=1.5,linestyle='--')
#         #plt.scatter(age_vec[interp_start::path_skip], curves_dsec_saved[interp_start::path_skip,any_min[j],c], marker='o', s=40, facecolor=plot_col[c], edgecolor='none')
#
#     plt.xlabel('Age [Myr]', fontsize=9)
#     plt.ylabel('Secondary Growth Rate', fontsize=9)
#     plt.title('interpolated growth rate histories dual', color=plot_col[c])
#     plt.xlim([0.5,5.0])



plt.savefig(outpath+"q_interp2d_dsec_alt.png",bbox_inches='tight')
plt.savefig(outpath+"z_interp2d_dsec_alt.eps",bbox_inches='tight')









#hack: 2D dsec curve FB!

fig=plt.figure(figsize=(16.0,12.0*2.0/3.0))
plt.subplots_adjust(hspace=0.4)

the_dashes = [6, 2]

ax=fig.add_subplot(3, 4, 1, frameon=True)
#plt.grid()
for c in [6, 7, 8, 9]:
    ax.fill_between(age_vec[interp_start:], curves_dsec_n_saved_d[interp_start:,14,c,1], curves_dsec_n_saved_d[interp_start:,14,c,2],facecolor=plot_col[c], linewidth=0.0, alpha=0.5)
plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Secondary Growth Rate', fontsize=9)
plt.title('Fe-Celadonite', fontsize=10)
plt.xlim([0.5,5.0])
# plt.legend( fontsize=10,bbox_to_anchor=(1.0, 1.4),ncol=1,columnspacing=0.0,labelspacing=0.0)


ax=fig.add_subplot(3, 4, 5, frameon=True)
#plt.grid()
for c in [6, 7, 8, 9]:
    ax.fill_between(age_vec[interp_start:], curves_dsec_n_saved_d[interp_start:,17,c,1], curves_dsec_n_saved_d[interp_start:,17,c,2],facecolor=plot_col[c], linewidth=0.0, alpha=0.5)
plt.xlabel('Age [Myr]', fontsize=9)
plt.ylabel('Secondary Growth Rate', fontsize=9)
plt.title('Hematite', fontsize=10)
plt.xlim([0.5,5.0])






plt.savefig(outpath+"q_interp2d_dsec_fb.png",bbox_inches='tight')
plt.savefig(outpath+"z_interp2d_dsec_fb.eps",bbox_inches='tight')











cont_cmap = cm.rainbow
n_cont = 41
cont_skip = 10
bar_shrink = 0.9
clabelpad = 0
xskip = 2
yskip = 1

sp1 = 3
sp2 = 4

#hack: FIG: alt_ind CONTOURS
print "2d_alt_vol contour"
fig=plt.figure(figsize=(12.0,9.0))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

alt_vol_data_contours = [0.0, 5.38, 4.7, 10.2]

#hack: interp param_nums
q_interp = (q_vec[:,6])*(3.14e7)
f = interpolate.interp1d(age_vec, (q_vec[:,6])*(3.14e7))
# f = interpolate.interp1d((q_vec[:,6])*(3.14e7), age_vec)
# print "q at 250kyr: ", f(0.25)
print "q at 500kyr: ", f(0.5)
print "q at 1000kyr: ", f(1.0)
print "q at 1500kyr: ", f(1.5)
print "q at 1500kyr: ", f(2.0)
print "q at 1500kyr: ", f(2.5)
print "q at 1500kyr: ", f(3.0)
# print "age of at 4.0q" , f(4.0)
age_cont_y_nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
age_cont_y_labels = f(age_cont_y_nums)
print age_cont_y_labels
age_cont_y_strings = []
for i in range(len(age_cont_y_nums)):
    age_cont_y_strings.append(str(age_cont_y_nums[i]) + " Myr")


### FIRST ROW, S D A B ALT_VOL MEAN SLOPES ###
x_cont = diff_nums
y_cont = param_nums
x_grid, y_grid = np.meshgrid(x_cont,y_cont)
x_grid = x_grid[:cont_y_param_max,:cont_x_diff_max]
y_grid = y_grid[:cont_y_param_max,:cont_x_diff_max]

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

cont_levels = np.linspace(min_all,max_all,num=n_cont,endpoint=True)

square_contour(sp1, sp2, 1, the_s, cb_title="s percent alt / Myr", xlab=1, ylab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

square_contour(sp1, sp2, 2, the_d, cb_title="value_alt_vol_mean_d", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

square_contour(sp1, sp2, 3, the_a, cb_title="value_alt_vol_mean_a", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

square_contour(sp1, sp2, 4, the_b, cb_title="value_alt_vol_mean_b", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)






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

cont_levels = np.linspace(min_all,max_all,num=n_cont,endpoint=True)

square_contour(sp1, sp2, 5, the_s, cb_title="value_alt_fe_mean", xlab=1, ylab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

square_contour(sp1, sp2, 6, the_d, cb_title="value_alt_fe_mean_d", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

square_contour(sp1, sp2, 7, the_a, cb_title="value_alt_fe_mean_a", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

square_contour(sp1, sp2, 8, the_b, cb_title="value_alt_fe_mean_b", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)




plt.savefig(in_path+dir_path[:-1]+"_alt_cont.png",bbox_inches='tight')
plt.savefig(in_path+"z_"+ dir_path[:-1]+"_alt_cont.eps",bbox_inches='tight')







print "2d_alt_vol contour lim"
fig=plt.figure(figsize=(14.0,10.0))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# alt_vol_data_contours = [0.0, 5.38, 4.7, 10.2]
#
# #hack: interp param_nums
# q_interp = (q_vec[:,6])*(3.14e7)
# f = interpolate.interp1d(age_vec, (q_vec[:,6])*(3.14e7))
# # f = interpolate.interp1d((q_vec[:,6])*(3.14e7), age_vec)
# # print "q at 250kyr: ", f(0.25)
# print "q at 500kyr: ", f(0.5)
# print "q at 1000kyr: ", f(1.0)
# print "q at 1500kyr: ", f(1.5)
# print "q at 1500kyr: ", f(2.0)
# print "q at 1500kyr: ", f(2.5)
# print "q at 1500kyr: ", f(3.0)
# # print "age of at 4.0q" , f(4.0)
# age_cont_y_nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# age_cont_y_labels = f(age_cont_y_nums)
# print age_cont_y_labels
# age_cont_y_strings = []
# for i in range(len(age_cont_y_nums)):
#     age_cont_y_strings.append(str(age_cont_y_nums[i]) + " Myr")


### FIRST ROW, S D A B ALT_VOL MEAN SLOPES ###
# x_cont = diff_nums
# y_cont = param_nums
# x_grid, y_grid = np.meshgrid(x_cont,y_cont)
# x_grid = x_grid[:cont_y_param_max,:cont_x_diff_max]
# y_grid = y_grid[:cont_y_param_max,:cont_x_diff_max]

the_s = value_alt_vol_mean[:cont_y_param_max,:cont_x_diff_max,0]
the_d = value_alt_vol_mean_d[:cont_y_param_max,:cont_x_diff_max,0]
# the_a = value_alt_vol_mean_a[:cont_y_param_max,:cont_x_diff_max,0]
# the_b = value_alt_vol_mean_b[:cont_y_param_max,:cont_x_diff_max,0]

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

square_contour(sp1, sp2, 1, the_s, cb_title="s percent alt / Myr", xlab=1, ylab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

square_contour(sp1, sp2, 2, the_d, cb_title="value_alt_vol_mean_d", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

# square_contour(sp1, sp2, 3, the_a, cb_title="value_alt_vol_mean_a", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)
#
# square_contour(sp1, sp2, 4, the_b, cb_title="value_alt_vol_mean_b", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)






the_s = value_alt_fe_mean[:cont_y_param_max,:cont_x_diff_max,0]
the_d = value_alt_fe_mean_d[:cont_y_param_max,:cont_x_diff_max,0]
# the_a = value_alt_fe_mean_a[:cont_y_param_max,:cont_x_diff_max,0]
# the_b = value_alt_fe_mean_b[:cont_y_param_max,:cont_x_diff_max,0]

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

square_contour(sp1, sp2, 5, the_s, cb_title="value_alt_fe_mean", xlab=1, ylab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

square_contour(sp1, sp2, 6, the_d, cb_title="value_alt_fe_mean_d", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

# square_contour(sp1, sp2, 7, the_a, cb_title="value_alt_fe_mean_a", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)
#
# square_contour(sp1, sp2, 8, the_b, cb_title="value_alt_fe_mean_b", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)




plt.savefig(in_path+dir_path[:-1]+"_alt_cont_lim.png",bbox_inches='tight')
plt.savefig(in_path+"z_"+ dir_path[:-1]+"_alt_cont_lim.eps",bbox_inches='tight')







sp1 = 3
sp2 = 4

all_params = 1

if all_params == 1:
    cont_x_diff_max = len(diff_strings) - 0
    cont_y_param_max = len(param_strings) - 0


#hack: FIG: dsec CONTOURS
print "2d_dsec contour"
fig=plt.figure(figsize=(12.0,9.0))
plt.subplots_adjust(wspace=0.5, hspace=0.5)


#hack: interp param_nums
q_interp = (q_vec[:,6])*(3.14e7)
f = interpolate.interp1d(age_vec, (q_vec[:,6])*(3.14e7))
# f = interpolate.interp1d((q_vec[:,6])*(3.14e7), age_vec)
# print "q at 250kyr: ", f(0.25)
print "q at 500kyr: ", f(0.5)
print "q at 1000kyr: ", f(1.0)
print "q at 1500kyr: ", f(1.5)
print "q at 1500kyr: ", f(2.0)
print "q at 1500kyr: ", f(2.5)
print "q at 1500kyr: ", f(3.0)
# print "age of at 4.0q" , f(4.0)
age_cont_y_nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
age_cont_y_labels = f(age_cont_y_nums)
print age_cont_y_labels
age_cont_y_strings = []
for i in range(len(age_cont_y_nums)):
    age_cont_y_strings.append(str(age_cont_y_nums[i]) + " Myr")


### FIRST ROW, S D A B ALT_VOL MEAN SLOPES ###
x_cont = diff_nums
y_cont = param_nums
x_grid, y_grid = np.meshgrid(x_cont,y_cont)
x_grid = x_grid[:cont_y_param_max,:cont_x_diff_max]
y_grid = y_grid[:cont_y_param_max,:cont_x_diff_max]


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

square_contour_min(sp1, sp2, 1, the_s, cb_title="s dsec rate "+secondary[the_min], xlab=1, ylab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels,the_cbar=1)

square_contour_min(sp1, sp2, 2, the_d, cb_title="d dsec rate", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

square_contour_min(sp1, sp2, 3, the_a, cb_title="a dsec rate", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)

square_contour_min(sp1, sp2, 4, the_b, cb_title="b dsec rate", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)





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
# cont_levels = np.linspace(min_all,max_all,num=n_cont,endpoint=True)
#
# square_contour(sp1, sp2, 5, the_s, cb_title="value_alt_fe_mean", xlab=1, ylab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)
#
# square_contour(sp1, sp2, 6, the_d, cb_title="value_alt_fe_mean_d", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)
#
# square_contour(sp1, sp2, 7, the_a, cb_title="value_alt_fe_mean_a", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)
#
# square_contour(sp1, sp2, 8, the_b, cb_title="value_alt_fe_mean_b", xlab=1, age_ticks=age_cont_y_strings, age_tick_labels=age_cont_y_labels)




plt.savefig(in_path+dir_path[:-1]+"_dsec.png",bbox_inches='tight')
plt.savefig(in_path+"z_"+ dir_path[:-1]+"_dsec.eps",bbox_inches='tight')
