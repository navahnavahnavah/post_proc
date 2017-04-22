# multi_select.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import math
import multiplot_data as mpd
from numpy.ma import masked_array
from mpl_toolkits.axes_grid1 import AxesGrid
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
# plt.rc('title', fontsize=8)
# plt.rc('ylabel', labelsize=8)
# plt.rc('xlabel', labelsize=8)


plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

path = 'output/revival/wip_plots/'
path_list = ['output/revival/multi_push_orhs500_16cells/o250orhs500w1000wrhs1000h100/', 'output/revival/multi_push_orhs400_16cells/o250orhs400w800wrhs600h100/',
'output/revival/multi_push_orhs300_16cells/o250orhs300w800wrhs1400h100/']
label_list = ['left oucrop: 1000m x 250 m, right outcrop: 1000m x 500m', 'left oucrop: 800m x 250 m, right outcrop: 600m x 400m',
 'left oucrop: 800m x 250 m, right outcrop: 1400m x 300m']
color_list = ['#CE1836', '#F85931', '#EDB92E', '#A3A948', '#009989']
param_w_list = [1000.0, 800.0, 800.0]

sed = np.abs(mpd.interp_s)
sed1 = np.abs(mpd.interp_b)
sed2 = np.abs(mpd.interp_s - mpd.interp_b)

print sed.shape

x = np.loadtxt(path_list[0] + 'x.txt',delimiter='\n')
y = np.loadtxt(path_list[0] + 'y.txt',delimiter='\n')
xg, yg = np.meshgrid(x[:],y[:])

ripTrans = np.transpose(mpd.ripSort)
hf_interp = np.zeros(200)
hf_interp = np.interp(x,ripTrans[0,:],ripTrans[1,:])

param_w = 600.0



heat_group = np.zeros([len(x),len(label_list)])
heat_error = np.zeros([len(label_list)])

c14_mean_group = np.zeros([len(x),len(label_list)])
c14_top_group = np.zeros([len(x),len(label_list)])

step = 10

for i in range(len(path_list)):
    print label_list[i]
    print i
    
    # load stuff
    uMat0 = np.loadtxt(path_list[i] + 'uMat.txt')*(3.14e7)#*10.0
    vMat0 = np.loadtxt(path_list[i] + 'vMat.txt')*(3.14e7)#*10.0
    psiMat0 = np.loadtxt(path_list[i] + 'psiMat.txt')
    hMat0 = np.loadtxt(path_list[i] + 'hMat.txt')
    permMat0 = np.log10(np.loadtxt(path_list[i] + 'permMat.txt'))
    maskP = np.loadtxt(path_list[i] + 'maskP.txt')
    lambdaMat = np.loadtxt(path_list[i] + 'lambdaMat.txt')
    c140 = np.loadtxt(path_list[i] + 'iso_c14.txt')
    
    
    u_last = uMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
    v_last = vMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
    psi_last = psiMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
    h_last = hMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
    perm_last = permMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
    c14_last = c140[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
    
    
    ############ HEAT MULTI SELECT ############

    sh = 10
    shc = 0
    
    heatVec = np.zeros(len(h_last[-1,:]))
    heatSed = np.zeros(len(h_last[-1,:]))
    heatBottom = np.zeros(len(h_last[-1,:]))
    for m in range(len(x)):
        heatBottom[m] = -1000.0*2.0*(h_last[2,m] - h_last[1,m])/(y[1]-y[0])
        for n in range(len(y)):
            if (maskP[n,m] == 25.0) or (maskP[n,m] == 50.0):
                heatVec[m] = -1000.0*lambdaMat[n-1,m]*(h_last[n+1,m] - h_last[n,m])/(y[1]-y[0])
                heatSed[m] = -1000.0*1.2*(h_last[n+1,m] - h_last[n,m])/(y[1]-y[0])
                
    heat_group[sh:-sh,i] = heatVec[sh:-sh]*np.min(sed1[sh:-sh])/(sed1[sh:-sh])

    print "heat flow error"
    heat_error[i] =  np.sum(np.abs(hf_interp[sh-shc:-sh-shc] - heat_group[sh:-sh,i]))/np.sum(np.abs(hf_interp[sh-shc:-sh-shc]))
    print heat_error[i]
    print " "
    
    
    
    
    ############ 14C MULTI SELECT ############
    
    varStep0 = c14_last
    varStep = np.zeros(c14_last.shape)

    for jj in range(len(x)-1):
        for j in range(len(y)-1):
                if varStep0[j,jj] > 1e-27:
                    varStep[j,jj] = 5730.0*np.log(varStep0[j,jj])/(-.693)
                if varStep0[j,jj] <= 1e-27:
                    varStep[j,jj] = 5730.0*np.log(1e-27)/(-.693)

    
    iso_col_mean = np.zeros(len(x))
    iso_col_top = np.zeros(len(x))
    varStepMask = np.zeros(varStep.shape)
    for jj in range(len(x)-1):
        for j in range(len(y)-1):
            if perm_last[j,jj] > -13 and maskP[j,jj] ==1.0:
                varStepMask[j,jj] = varStep[j,jj]
                
    for jj in range(len(x)-1):
        c14_mean_group[jj,i] = np.sum(varStepMask[:,jj])/np.count_nonzero(varStepMask[:,jj])
        for j in range(len(y)-2):
            if perm_last[j,jj] > -13.0 and perm_last[j+1,jj] <= -15.0:
                c14_top_group[jj,i] = varStepMask[j,jj]








fig=plt.figure()
ax1=fig.add_subplot(2,1,1)
    


plt.plot(mpd.steinX1[::2],mpd.steinY1[::2], 'k', linestyle="-", lw=1, label='Stein 2004 7.9 m/yr')
plt.plot(mpd.steinX2[::2],mpd.steinY2[::2], 'grey', linestyle="-", lw=1, label='Stein 2004 4.9 m/yr')
# p = plt.plot(x,hf_interp,'k-',linewidth=1.0,label='HF interpolation')
for i in range(len(path_list)):
    p = plt.plot(x[sh+shc:-sh+shc]-1200.0,heat_group[sh:-sh,i],color_list[i],linewidth=1,label=label_list[i])


plt.scatter(mpd.ripXdata,mpd.ripQdata,s=15,c='k',label='data')
p = plt.plot(x,heatBottom,'b',linewidth=1.0,label='Conduction')

ax1.grid(which='major', alpha=1.0)
plt.ylim([-100.0,800.0])
plt.xlim([np.min(x),26000.0])
plt.legend(fontsize=7,bbox_to_anchor=(0.5, 1.45), loc='upper center',ncol=2)
plt.ylabel('heat flow [mW/m^2]')



ax1=fig.add_subplot(2,1,2)
plt.scatter([param_w+3300.0, param_w+6700.0, param_w+14600.0, param_w+21000.0, param_w+21000.0], [1000.0, 6210.0, 9930.0, 7720.0, 7810.0],c='k',s=30,edgecolor='k',label='data',zorder=10)

for i in range(len(path_list)):
    print i
    plt.plot(x-param_w_list[i],c14_top_group[:,i],color_list[i],label=label_list[i])
    #plt.plot(x-param_w_list[i],c14_mean_group[:,i],color_list[i],linestyle='-',label=label_list[i])


# plt.legend(fontsize=7,bbox_to_anchor=(0.5, 1.3), loc='upper center',ncol=2)

ax1.grid(which='major', alpha=1.0)
plt.xlim([np.min(x),26000.0])
plt.ylim([0.0,100000.0])

plt.ylabel('fluid 14C age')
plt.xlabel('distance from outcrop edge [m]')

plt.subplots_adjust(top=0.8)

plt.savefig(path+'hf.eps')
