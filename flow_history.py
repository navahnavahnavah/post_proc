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

plot_col = ['#000000', '#940000', '#d26618', '#dfa524', '#9ac116', '#139a59', '#35b5aa', '#0740d2', '#7f05d4', '#b100de']

outpath = "../output/revival/winter_basalt_box/"


def square_contour(sp1, sp2, sp, cont_block, cb_title="", xlab=0, ylab=0, age_ticks=[1.0, 2.0], age_tick_labels=[1.0, 2.0]):
    ax1=fig.add_subplot(sp1, sp2, sp, frameon=True)

    # print age_ticks
    # print age_tick_labels

    if xlab == 1:
        plt.xlabel('log10(mixing time [years])', fontsize=9)
    if ylab == 1:
        plt.ylabel('discharge q [m/yr]', fontsize=9)

    pCont = ax1.contourf(x_grid,y_grid,cont_block, levels=cont_levels, cmap=cont_cmap, antialiased=True, linewidth=0.0)
    for c in pCont.collections:
        c.set_edgecolor("face")

    plt.xticks(diff_nums[:cont_x_diff_max:xskip],diff_strings[::xskip], fontsize=8)
    plt.yticks(param_nums[:cont_y_param_max:yskip],param_strings[::yskip], fontsize=8)

    ax2 = ax1.twinx()
    ax2.set_ylabel('age [myr]', fontsize=9)
    # ax2.set_yticks([1.0, 2.0],["1.0", "2.0"])
    plt.yticks(age_tick_labels, age_ticks, fontsize=8)
    plt.ylim([0.5,4.5])



    # cbar = fig.colorbar(pCont, orientation='horizontal',shrink=bar_shrink)
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_ticks(cont_levels[::cont_skip])
    # cbar.ax.set_xlabel(cb_title,fontsize=9,labelpad=clabelpad)
    # cbar.solids.set_edgecolor("face")

    bbox = ax2.get_position()
    cax = fig.add_axes([bbox.xmin+0.0, bbox.ymin-0.05, bbox.width*1.0, bbox.height*0.05])
    cbar = plt.colorbar(pCont, cax = cax,orientation='horizontal')
    cbar.set_ticks(cont_levels[::cont_skip])
    cbar.ax.tick_params(labelsize=7)
    #plt.title('CaCO3 at end',fontsize=9)
    cbar.solids.set_edgecolor("face")




    # plt.contour(x_grid, y_grid, cont_block, levels=alt_vol_data_contours, colors='w', linewidth=3.0)
    return square_contour













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


txt_labels = ['sed 5m/Myr, 500m basement', 'sed 50m/Myr, 500m basement', 'sed 5m/Myr, 100m basement', 'sed 50m/Myr, 100m basement', 'sed 100m fixed, 500m basement', 'sed 200m fixed, 500m basement', 'sed 50m/yr, 500m basement, dx fixed at 10km', 'sed 50m/yr, 500m basement, dx fixed at 20km', 'sed 50m/yr, 500m basement, dx fixed at 100km', 'sed 100m/yr, 500m basement, dx fixed at 100km']


# generate qm/ql (hfs_vec)
hfs_vec = np.zeros([len(steps),n_hist])
for i in range(nsteps):
    # n_hist = 0 : fisher2000, sediment 5m/Myr, basement 500m
    hfs_vec[i,0] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,0] = 5.0 * age_vec[i]
    dx_vec[i,0] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[0] = 500.0

    # n_hist = 1 : fisher2000, sediment 50m/Myr, basement 500m
    hfs_vec[i,1] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,1] = 50.0 * age_vec[i]
    dx_vec[i,1] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[1] = 500.0

    # n_hist = 2 : fisher2000, sediment 5m/Myr, basement 100m
    hfs_vec[i,2] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,2] = 5.0 * age_vec[i]
    dx_vec[i,2] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[2] = 100.0

    # n_hist = 3 : fisher2000, sediment 50m/Myr, basement 100m
    hfs_vec[i,3] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,3] = 50.0 * age_vec[i]
    dx_vec[i,3] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[3] = 100.0

    # n_hist = 4 : sediment fixed at 100m, basement 500
    hfs_vec[i,4] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,4] = 100.0
    dx_vec[i,4] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[4] = 500.0

    # n_hist = 5 : sediment fixed at 200m, basement 500
    hfs_vec[i,5] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,5] = 200.0
    dx_vec[i,5] = (5000.0 + 500.0*age_vec[i])/2.0
    param_hb[5] = 500.0

    # n_hist = 6 : sediment 50m/Myr, dx fixed at 10km
    hfs_vec[i,6] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,6] = 50.0 * age_vec[i]
    dx_vec[i,6] = 10000.0 #(5000.0 + 500.0*age_vec[i])/2.0
    param_hb[6] = 500.0

    # n_hist = 7 : sediment 50m/Myr, dx fixed at 20km
    hfs_vec[i,7] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,7] = 50.0 * age_vec[i]
    dx_vec[i,7] = 20000.0 #(5000.0 + 500.0*age_vec[i])/2.0
    param_hb[7] = 500.0

    # n_hist = 8 : sediment 50m/Myr, dx fixed at 100km
    hfs_vec[i,8] = 0.43 + 0.0085*age_vec[i]
    hs_vec[i,8] = 50.0 * age_vec[i]
    dx_vec[i,8] = 100000.0 #(5000.0 + 500.0*age_vec[i])/2.0
    param_hb[8] = 500.0

    # n_hist = 9 : sediment 100m/Myr, dx fixed at 100km
    hfs_vec[i,9] = 0.43 + 0.0085*age_vec[i]
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


#todo: FIG: fisher + variations plot
the_lw = 2.0
m_size = 50
fig=plt.figure(figsize=(12.0,5.0))

ax1=fig.add_subplot(1, 3, 2, frameon=True)


plt.plot(age_vec,q_vec_log10[:,0], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::100],q_vec_log10[::100,0], marker='o', zorder=5, facecolor='k', s=m_size, label=txt_labels[0])

plt.plot(age_vec,q_vec_log10[:,1], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::100],q_vec_log10[::100,1], marker='s', zorder=5, facecolor='k', s=m_size, label=txt_labels[1])

plt.plot(age_vec,q_vec_log10[:,2], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::100],q_vec_log10[::100,2], marker='o', zorder=5, facecolor='none', s=m_size, label=txt_labels[2])

plt.plot(age_vec,q_vec_log10[:,3], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::100],q_vec_log10[::100,3], marker='s', zorder=5, facecolor='none', s=m_size, label=txt_labels[3])

for jj in range(4,10):
    plt.plot(age_vec,q_vec_log10[:,jj], color=plot_col[jj-3], label=txt_labels[jj], lw=the_lw)

plt.ylabel('log10(q [m/s])')
#plt.legend(fontsize=8,bbox_to_anchor=(1.6, 1.0),ncol=1,columnspacing=0.1)

plt.xlim([np.min(age_vec),np.max(age_vec)])





ax1=fig.add_subplot(1, 3, 3, frameon=True)


plt.plot(age_vec,q_vec_log10_myr[:,0], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::100],q_vec_log10_myr[::100,0], marker='o', zorder=5, facecolor='k', s=m_size, label=txt_labels[0])

plt.plot(age_vec,q_vec_log10_myr[:,1], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::100],q_vec_log10_myr[::100,1], marker='s', zorder=5, facecolor='k', s=m_size, label=txt_labels[1])

plt.plot(age_vec,q_vec_log10_myr[:,2], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::100],q_vec_log10_myr[::100,2], marker='o', zorder=5, facecolor='none', s=m_size, label=txt_labels[2])

plt.plot(age_vec,q_vec_log10_myr[:,3], color=plot_col[0], lw=the_lw)
plt.scatter(age_vec[::100],q_vec_log10_myr[::100,3], marker='s', zorder=5, facecolor='none', s=m_size, label=txt_labels[3])

for jj in range(4,10):
    plt.plot(age_vec,q_vec_log10_myr[:,jj], color=plot_col[jj-3], label=txt_labels[jj], lw=the_lw)

plt.yticks([-1.0, 0.0, 1.0, 2.0], [0.1, 1.0, 10.0, 100.0])

plt.ylabel('q [m/yr]')
plt.legend(fontsize=8,bbox_to_anchor=(1.7, 1.0),ncol=1,columnspacing=0.1)

plt.xlim([np.min(age_vec),np.max(age_vec)])



#plt.plot()

ax1.set_xlabel('age [Myr]')





ax1=fig.add_subplot(1, 3, 1, frameon=True)
ax1.grid()

# plt.plot(age_vec,q_vec[:,0], color=plot_col[0], lw=the_lw)
# plt.scatter(age_vec[::10],q_vec_log10[::10,0], marker='o', zorder=5, facecolor='k', s=m_size, label=txt_labels[0])
#
# plt.plot(age_vec,q_vec[:,1], color=plot_col[0], lw=the_lw)
# plt.scatter(age_vec[::10],q_vec_log10[::10,1], marker='s', zorder=5, facecolor='k', s=m_size, label=txt_labels[1])
#
# plt.plot(age_vec,q_vec[:,2], color=plot_col[0], lw=the_lw)
# plt.scatter(age_vec[::10],q_vec_log10[::10,2], marker='o', zorder=5, facecolor='none', s=m_size, label=txt_labels[2])
#
# plt.plot(age_vec,q_vec[:,3], color=plot_col[0], lw=the_lw)
# plt.scatter(age_vec[::10],q_vec_log10[::10,3], marker='s', zorder=5, facecolor='none', s=m_size, label=txt_labels[3])

for jj in [6]:
    plt.plot(age_vec[50:],q_vec[50:,jj]*(3.14e7), color=plot_col[jj-3], label=txt_labels[jj], lw=the_lw)

plt.ylabel('q [m/yr]')
#plt.legend(fontsize=8,bbox_to_anchor=(1.6, 1.0),ncol=1,columnspacing=0.1)

#plt.xlim([np.min(age_vec[100:]),np.max(age_vec)])


plt.savefig(outpath+"q_history.png",bbox_inches='tight')







#todo: path stuff
in_path = "../output/revival/winter_basalt_box/"



#todo: make 2d alt index grid arrays

param_strings = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5']
param_nums = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

diff_strings = ['2.00', '2.25', '2.50', '2.75', '3.00', '3.25', '3.50', '3.75', '4.00', '4.25', '4.50']
diff_nums = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5]


n_grids = 3

value_alt_vol_mean = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_vol_mean_d = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_vol_mean_a = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_vol_mean_b = np.zeros([len(param_strings),len(diff_strings),n_grids])

value_alt_fe_mean = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_fe_mean_d = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_fe_mean_a = np.zeros([len(param_strings),len(diff_strings),n_grids])
value_alt_fe_mean_b = np.zeros([len(param_strings),len(diff_strings),n_grids])

#todo: LOAD IN 2d alt index grids

dir_path = "y_group_d/"

value_alt_vol_mean[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean.txt')
value_alt_vol_mean_d[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean_d.txt')
value_alt_vol_mean_a[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean_a.txt')
value_alt_vol_mean_b[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_vol_mean_b.txt')

value_alt_fe_mean[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean.txt')
value_alt_fe_mean_d[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean_d.txt')
value_alt_fe_mean_a[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean_a.txt')
value_alt_fe_mean_b[:,:,0] = np.loadtxt(in_path + dir_path + 'value_alt_fe_mean_b.txt')

cont_x_diff_max = len(diff_strings) - 2
cont_y_param_max = len(param_strings) - 0






cont_cmap = cm.rainbow
n_cont = 41
cont_skip = 10
bar_shrink = 0.9
clabelpad = 0
xskip = 2
yskip = 1

sp1 = 3
sp2 = 4


#hack: 2d alt_vol CONTOUR
print "2d_alt_vol contour"
fig=plt.figure(figsize=(12.0,10.0))
plt.subplots_adjust(wspace=0.4, hspace=0.5)

alt_vol_data_contours = [0.0, 5.38]

#hack: interp param_nums
q_interp = (q_vec[:,6])*(3.14e7)
f = interpolate.interp1d(age_vec, (q_vec[:,6])*(3.14e7))
# f = interpolate.interp1d((q_vec[:,6])*(3.14e7), age_vec)
# print "q at 250kyr: ", f(0.25)
print "q at 500kyr: ", f(0.5)
print "q at 1000kyr: ", f(1.0)
print "q at 1500kyr: ", f(1.5)
# print "age of at 4.0q" , f(4.0)
age_cont_y_nums = [0.5, 1.0, 1.5]
age_cont_y_labels = f(age_cont_y_nums)
print age_cont_y_labels


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

square_contour(sp1, sp2, 1, the_s, cb_title="value_alt_vol_mean", xlab=1, ylab=1, age_ticks=age_cont_y_nums, age_tick_labels=age_cont_y_labels)

square_contour(sp1, sp2, 2, the_d, cb_title="value_alt_vol_mean_d", xlab=1)

square_contour(sp1, sp2, 3, the_a, cb_title="value_alt_vol_mean_a", xlab=1)

square_contour(sp1, sp2, 4, the_b, cb_title="value_alt_vol_mean_b", xlab=1)






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

square_contour(sp1, sp2, 5, the_s, cb_title="value_alt_fe_mean", xlab=1, ylab=1)

square_contour(sp1, sp2, 6, the_d, cb_title="value_alt_fe_mean_d", xlab=1)

square_contour(sp1, sp2, 7, the_a, cb_title="value_alt_fe_mean_a", xlab=1)

square_contour(sp1, sp2, 8, the_b, cb_title="value_alt_fe_mean_b", xlab=1)




plt.savefig(in_path+dir_path[:-1]+"_alt_cont.png",bbox_inches='tight')
