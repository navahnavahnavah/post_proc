# revived_JDF.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
# import streamplot as sp
# import multiplot_data as mpd
import heapq
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams['axes.titlesize'] = 12

plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

# plot_col = ['#940000', '#d26618', '#dfa524', '#9ac116', '#139a31', '#35b5aa', '#0740d2', '#6c05d4', '#9e00de', '#e287f7']
plot_col = ['#801515', '#c90d0d', '#d26618', '#dfa524', '#cdeb14', '#7d9d10', '#1ff675', '#139a72', '#359ab5', '#075fd2', '#151fa4', '#3c33a3', '#7f05d4', '#b100de', '#ff8ac2', '#ff8ac2']


#todo: parameters

#steps = 400
steps = 10
minNum = 57
ison=10000
trace = 0
chem = 1
iso = 0
cell = 5

#poop: path
# sub_dir = "ao_0.50/"
print "COMMAND LINE ARGUMENTS " + sys.argv[1]
sub_dir = "ao_" + str(sys.argv[1]) + "/"

#outpath = "../output/revival/local_fp_output/oc_output/oc_k_11_s_100_h_200_ts/par_q_5.0/" + sub_dir

outpath = "../output/revival/local_fp_output/nov_fp_tests/" + sub_dir
path = outpath
param_w = 300.0
param_w_rhs = 200.0

domain_x_ticks = np.linspace(0.0, 90000.0, 10)
domain_x_tick_labels = np.linspace(0, 90, 10)

domain_y_ticks = np.linspace(-1200.0, 0.0, 4)


# load output
x0 = np.loadtxt(path + 'x.txt',delimiter='\n')
y0 = np.loadtxt(path + 'y.txt',delimiter='\n')


# format plotting geometry
x=x0
y=y0

asp = np.abs(np.max(x)/np.min(y))
print asp
bitsx = len(x)
bitsy = len(y)
restart = 1
print "bitsy" , bitsy


dx = float(np.max(x))/float(bitsx)
dy = np.abs(float(np.max(np.abs(y)))/float(bitsy))

xCell = x0[1::cell]
# xCell = xCell[:-1]
# xCell = np.append(xCell, np.max(xCell))
yCell = y0[1::cell]
# yCell = np.append(yCell, np.max(yCell))

xg, yg = np.meshgrid(x[:],y[:])
xgCell, ygCell = np.meshgrid(xCell[:],yCell[:])

# mask = np.loadtxt(path + 'mask.txt')
maskP = np.loadtxt(path + 'maskP.txt')
psi0 = np.loadtxt(path + 'psiMat.txt')
perm = np.loadtxt(path + 'permeability.txt')

perm = np.log10(perm)

first_flow = np.argmax(perm[:,-1]>-16.0)
cells_flow = np.argmax(perm[first_flow:,-1]<-16.0)
print "cells flow count: " , cells_flow
cells_sed = np.argmax(perm[first_flow+cells_flow:,-1]>-16.0)
print "cells sed count: " , cells_sed
cells_above = len(y) - cells_sed - cells_flow - first_flow
print "cells above count: " , cells_above
print perm[:,-1]

temp0 = np.loadtxt(path + 'hMat.txt')
temp0 = temp0 - 273.0
# u0 = np.loadtxt(path + 'uMat.txt')
# v0 = np.loadtxt(path + 'vMat.txt')
# lambdaMat = np.loadtxt(path + 'lambdaMat.txt')

u_ts = np.zeros([steps])

# lam = np.loadtxt(path + 'lambdaMat.txt')


def cut(geo0,index):
    #geo_cut = geo0[(index*len(y0)/cell):(index*len(y0)/cell+len(y0)/cell),:]
    geo_cut = geo0[:,(index*len(x0)):(index*len(x0)+len(x0))]
    geo_cut = np.append(geo_cut, geo_cut[-1:,:], axis=0)
    geo_cut = np.append(geo_cut, geo_cut[:,-1:], axis=1)
    return geo_cut

def cut_chem(geo0,index):
    geo_cut_chem = geo0[:,(index*len(xCell)):(index*len(xCell)+len(xCell))]
    return geo_cut_chem

def chemplot(varMat, varStep, sp1, sp2, sp3, contour_interval,cp_title, ditch):
    if ditch==0:
        contours = np.linspace(np.min(varMat),np.max(varMat),5)
    if ditch==1:
        contours = np.linspace(np.min(varMat[varMat>0.0]),np.max(varMat),5)
        # varStep[varStep == 0.0] = None
        #contours = np.linspace(4.8,np.max(varStep),20)
    if ditch==2:
        contours = np.linspace(np.min(varMat),np.max(varMat[varMat<varStep[bitsy-25,bitsx/2]])/5.0,20)
        #contours = np.linspace(contours[0],contours[-1],20)
    ax1=fig.add_subplot(sp1,sp2,sp3, aspect=asp,frameon=False)
    #pGlass = plt.contourf(xCell,yCell,varStep,contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0,antialiased=True)
    if ditch==0:
        pGlass = plt.pcolor(xCell,yCell,varStep,cmap=cm.rainbow,vmin=np.min(contours), vmax=np.max(contours))
    if ditch==1:
        pGlass = plt.pcolor(xCell,yCell,varStep,cmap=cm.rainbow,vmin=np.min(contours), vmax=np.max(contours))
    p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([0.5]))
    plt.xticks([])
    plt.yticks([])
    plt.ylim([np.min(yCell),0.])
    #cMask = plt.contourf(xg,yg,maskP,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
    plt.title(cp_title,fontsize=10)
    cbar = plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::contour_interval])
    fig.set_tight_layout(True)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")
    pGlass.set_edgecolor("face")
    return chemplot


#todo: FIG: plot Q_lith
q_lith = np.zeros([len(x),10])
starts = np.linspace(0.25,0.75,11,endpoint=True)
# starts[0] = 0.50
# starts[1] = 0.75
# starts[2] = 1.00
# starts[3] = 1.25
# starts[4] = 1.50
# starts[5] = 1.75
# starts[6] = 2.00

for i in range(len(x)):
    for j in range(10):
        q_lith[i,j] = 0.5 / (((starts[j]+0.035*(x[1]-x[0])*i/1000.0)**0.5))


fig=plt.figure(figsize=(12.0,12.0))
ax=fig.add_subplot(2, 2, 1, frameon=True)

for j in range(6):
    plt.plot(x, q_lith[:,j], c=plot_col[j], label='start with '+str(starts[j]), lw=1.5)

plt.legend(bbox_to_anchor=(1.0,1.0),fontsize=8)
plt.savefig(outpath+'jdf_q_lith.png',bbox_inches='tight')


# delta = np.zeros(lambdaMat.shape)

conv_mean_qu = 0.0
conv_max_qu = 0.0
conv_mean_psi = 0.0
conv_max_psi = 0.0
conv_tot_hf = 0.0
conv_count = 0

bar_bins = 6

for i in range(0,steps,1):
    print " "
    print " "
    print "step =", i

    # single time slice matrices
    psi = psi0[:,i*len(x):((i)*len(x)+len(x))]
    # rho = rho0[:,i*len(x):((i)*len(x)+len(x))]
    # perm = perm0[:,i*len(x):((i)*len(x)+len(x))]
    temp = temp0[:,i*len(x):((i)*len(x)+len(x))]
    # u = u0[:,i*len(x):((i)*len(x)+len(x))]
    # v = v0[:,i*len(x):((i)*len(x)+len(x))]

    #poop: flow layer mean temp
    print maskP.shape
    print perm.shape
    flow_layer_mean = np.mean(temp[(perm>=-13.0) & (maskP>0.0)])
    print "FLOW LAYER MEAN" , flow_layer_mean

    # if i == steps-1:
    if i > 0:
        t_col_mean = np.zeros(len(x))
        t_col_bottom = np.zeros(len(x))
        t_col_top = np.zeros(len(x))

        for j in range(len(x)):
        #for j in range(20):
            temp_temp = temp[:,j]
            temp_perm = perm[:,j]
            temp_maskP = maskP[:,j]
            t_col_mean[j] = np.mean(temp_temp[(temp_perm>=-13.0) & (temp_maskP>0.0)])
            t_col_bottom[j] = np.max(temp_temp[(temp_perm>=-13.0) & (temp_maskP>0.0)])
            t_col_top[j] = np.min(temp_temp[(temp_perm>=-13.0) & (temp_maskP>0.0)])


        #todo: jdf_t_mean.png
        fig=plt.figure(figsize=(12.0,12.0))
        ax=fig.add_subplot(2, 2, 1, frameon=True)

        plt.plot(x,t_col_mean, label='t_col_mean', c='#d69a00',lw=2)
        plt.plot(x,t_col_bottom, label='t_col_bottom', c='r',lw=2)
        plt.plot(x,t_col_top, label='t_col_top', c='b',lw=2)

        plt.legend(fontsize=8,loc='best')
        plt.savefig(outpath+'jdf_t_mean'+str(i)+'.png',bbox_inches='tight')




        #todo: jdf_t_lat.png
        fig=plt.figure(figsize=(12.0,12.0))
        ax=fig.add_subplot(2, 2, 1, frameon=True)


        for j in range(cells_flow):

            plt.plot(x,temp[bitsy-cells_sed-cells_flow-cells_above+j,:], label=str(j),c=plot_col[j],lw=2)
        plt.title(sub_dir)
        plt.plot(x,temp[bitsy-cells_sed-cells_flow-cells_above-4,:], label=str(j),c='k',lw=1,linestyle='--')
        plt.plot(x,temp[bitsy-cells_sed-cells_flow-cells_above-3,:], label=str(j),c='k',lw=1,linestyle='--')
        plt.plot(x,temp[bitsy-cells_sed-cells_flow-cells_above-2,:], label=str(j),c='k',lw=1,linestyle='--')
        plt.plot(x,temp[bitsy-cells_sed-cells_flow-cells_above-1,:], label=str(j),c='k',lw=1,linestyle='--')
        plt.plot(x,temp[bitsy-cells_sed-cells_above,:], label=str(j),c='c',lw=1,linestyle='--')
        plt.plot(x,temp[bitsy-cells_sed-cells_above+1,:], label=str(j),c='c',lw=1,linestyle='--')
        plt.plot(x,temp[bitsy-cells_sed-cells_above+2,:], label=str(j),c='c',lw=1,linestyle='--')
        plt.plot(x,temp[bitsy-cells_sed-cells_above+3,:], label=str(j),c='c',lw=1,linestyle='--')

        plt.ylim([0.0,140.0])
        plt.legend(fontsize=8,loc='best')




        ax=fig.add_subplot(2, 2, 2, frameon=True)

        plt.title(sub_dir)
        plt.plot(x,-1.2*(temp[bitsy-cells_above,:]-temp[bitsy-cells_above-1,:])/25.0, label=str(j),c='k',lw=1,linestyle='--')
        plt.plot(x,-1.2*(temp[bitsy-cells_above-1,:]-temp[bitsy-cells_above-2,:])/25.0, label=str(j),c='r',lw=1,linestyle='--')
        plt.plot(x,-1.2*(temp[bitsy-cells_above-2,:]-temp[bitsy-cells_above-3,:])/25.0, label=str(j),c='b',lw=2,linestyle='--')
        plt.ylim([0.0,0.5])




        plt.savefig(outpath+'jdf_t_lat'+str(i)+'.png',bbox_inches='tight')




    # #todo: sed_thick.png
    # cap1 = int((param_w/50.0)) + 4
    # cap2 = int((param_w_rhs/50.0)) + 4
    #
    #
    # colMax = np.zeros(len(x))
    # for n in range(cap1,len(x)-cap2):
    #     cmax = np.max(u[:,n])*(3.14e7)
    #     cmin = np.min(u[:,n])*(3.14e7)
    #     if np.abs(cmax) > np.abs(cmin):
    #         colMax[n] = cmax
    #     if np.abs(cmax) < np.abs(cmin):
    #         colMax[n] = cmin
    #     #colMax[n] = cmax
    #     #print colMax[n]
    # colMean = np.sum(colMax)/len(x[cap1:-cap2])
    # print u_ts.shape
    # u_ts[i] = colMean
    #
    #
    #
    # fig=plt.figure()
    # # plt.plot(mpd.interp_s-mpd.interp_b)
    # plt.savefig(outpath+'sed_thick.png',bbox_inches='tight')





    #todo: FIG: jdf_i.png


    # fig=plt.figure(figsize=(10.0,6.0))
    #
    # y_limit = 2*len(y)/4
    #
    #
    # # temp plot
    # varStep = temp
    # varMat = varStep
    # contours = np.linspace(np.min(varStep),np.max(varStep),30)
    #
    # ax1=fig.add_subplot(2,1,2, aspect=asp/2.0,frameon=False)
    # pGlass = plt.contourf(x, y[y_limit:], varStep[y_limit:,:], 30, cmap=cm.rainbow, alpha=1.0,color='#444444',antialiased=True)
    # p = plt.contour(xg[y_limit:,:],yg[y_limit:,:],perm[y_limit:,:],[-14.9],colors='black',linewidths=np.array([1.5]))
    # CS = plt.contour(xg[y_limit:,:], yg[y_limit:,:], psi[y_limit:,:], 8, colors='black',linewidths=np.array([0.5]))
    # plt.xticks(domain_x_ticks,domain_x_tick_labels)
    # for c in pGlass.collections:
    #     c.set_edgecolor("face")
    # #cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    #
    # cbar= plt.colorbar(pGlass, orientation='horizontal')
    # cbar.set_ticks(np.linspace(np.min(varStep[y_limit:,:]),np.max(varStep[y_limit:,:]),num=bar_bins,endpoint=True))
    # cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
    # cbar.solids.set_edgecolor("face")
    #
    #
    #
    # # u velocity plot
    # varMat = u*(3.14e7)#ca0
    # varStep = u*(3.14e7)#ca
    # contours = np.linspace(np.min(varMat),np.max(varMat),10)
    #
    # ax1=fig.add_subplot(2,2,1, aspect=asp/4.0,frameon=False)
    # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    # plt.xticks(domain_x_ticks,domain_x_tick_labels)
    # plt.yticks(domain_y_ticks)
    # for c in pGlass.collections:
    #     c.set_edgecolor("face")
    #
    # cbar= plt.colorbar(pGlass, orientation='horizontal')
    # cbar.set_ticks(np.linspace(np.min(varStep),np.max(varStep),num=bar_bins,endpoint=True))
    # cbar.ax.set_xlabel('u [m/yr]')
    # cbar.solids.set_edgecolor("face")
    #
    #
    # # v velocity plot
    # varMat = v*(3.14e7)#dic0
    # varStep = v*(3.14e7)#dic
    # contours = np.linspace(np.min(varMat),np.max(varMat),10)
    #
    # ax1=fig.add_subplot(2,2,2, aspect=asp/4.0,frameon=False)
    # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    # plt.xticks(domain_x_ticks,domain_x_tick_labels)
    # plt.yticks(domain_y_ticks)
    # for c in pGlass.collections:
    #     c.set_edgecolor("face")
    #
    # cbar= plt.colorbar(pGlass, orientation='horizontal')
    # cbar.set_ticks(np.linspace(np.min(varStep),np.max(varStep),num=bar_bins,endpoint=True))
    # cbar.ax.set_xlabel('v [m/yr]')
    # cbar.solids.set_edgecolor("face")
    #
    # plt.savefig(outpath+'jdf_'+str(i+restart)+'.png',bbox_inches='tight')







    #hack: figure for paper

    fig=plt.figure(figsize=(30.0,12.0))

    y_limit = len(y)/2

    xn = len(x)
    lim_a = 0.0
    lim_b = 6000.0
    lim_a0 = int(lim_a/(x[1]-x[0]))
    lim_b0 = int(lim_b/(x[1]-x[0]))
    lim_u = 5*len(y)/12
    lim_o = len(y)

    aspSQ = asp/80.0
    aspZ = asp


    # temp plot
    varStep = temp
    varMat = varStep
    contours = np.linspace(np.min(varStep),np.max(varStep),30)

    ax1=fig.add_subplot(2,1,2, aspect=asp/2.0,frameon=False)
    pGlass = plt.contourf(x, y[y_limit:], varStep[y_limit:,:], 30, cmap=cm.rainbow, alpha=1.0,color='#444444',antialiased=True)
    p = plt.contour(xg[y_limit:,:],yg[y_limit:,:],perm[y_limit:,:],[-14.9],colors='black',linewidths=np.array([1.5]))
    CS = plt.contour(xg[y_limit:,:], yg[y_limit:,:], psi[y_limit:,:], 8, colors='black',linewidths=np.array([0.5]))
    plt.xticks(domain_x_ticks,domain_x_tick_labels)
    for c in pGlass.collections:
        c.set_edgecolor("face")
    #cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))


    ### NOVEMBER UPDATE
    cbar= plt.colorbar(pGlass, orientation='horizontal')
    cbar.set_ticks(np.linspace(np.min(varStep[y_limit:,:]),np.max(varStep[y_limit:,:]),num=bar_bins,endpoint=True))
    # cbar.set_ticks(0.0,100.0,num=bar_bins,endpoint=True))
    cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
    cbar.solids.set_edgecolor("face")





    varMat = temp[lim_u:lim_o,lim_a0:lim_b0]
    varStep = temp[lim_u:lim_o,lim_a0:lim_b0]
    contours = np.linspace(np.min(varStep),np.max(varStep),20)

    ax1=fig.add_subplot(2,2,1, aspect=aspSQ*3.0,frameon=False)
    pGlass = plt.contourf(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep, 40, cmap=cm.rainbow,vmin = 0.0,vmax=60.0)
    CS = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0], yg[lim_u:lim_o,lim_a0:lim_b0], psi[lim_u:lim_o,lim_a0:lim_b0], 8, colors='black',linewidths=np.array([0.5]))
    for c in pGlass.collections:
        c.set_edgecolor("face")
    p = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0],yg[lim_u:lim_o,lim_a0:lim_b0],perm[lim_u:lim_o,lim_a0:lim_b0],[-14.9,-15.0,-16.0,-13.5],colors='black',linewidths=np.array([1.5]))
    cMask = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0],yg[lim_u:lim_o,lim_a0:lim_b0],maskP[lim_u:lim_o,lim_a0:lim_b0],[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))

    plt.xlim([lim_a,lim_b])
    plt.ylim([7*np.min(y)/12,0.])
    cbar = plt.colorbar(pGlass,orientation='horizontal', fraction=0.046)
    cbar.set_ticks(np.linspace(0.0,80.0,num=9,endpoint=True))
    # cbar.set_ticks(np.linspace(np.min(temp),temp_max,5))
    # cbar.set_clim(np.min(temp), temp_max)
    cbar.solids.set_edgecolor("face")
    plt.title('LEFT OUTCROP new')


    ax=fig.add_subplot(2, 2, 2, frameon=True)
    plt.plot(varStep[:,::10],y[lim_u:lim_o])
    plt.plot(varStep[:,-5:],y[lim_u:lim_o],color='k',linewidth=2.0)
    plt.xlim([0.0,70.0])



    # varMat = temp[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
    # varStep = temp[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    #
    # ax1=fig.add_subplot(2,2,2, aspect=aspSQ/1000,frameon=False)
    # pGlass = plt.contourf(x[xn-lim_b0:xn-lim_a0]/1000, y[lim_u:lim_o], varStep, cmap=cm.rainbow,vmin = np.min(temp),vmax=180)
    # CS = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0]/1000, yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0], psi[lim_u:lim_o,xn-lim_b0:xn-lim_a0], 8, colors='black',linewidths=np.array([0.5]))
    # p = plt.contour(xg/1000,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    # cMask = plt.contour(xg/1000,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    #
    # plt.xlim([(np.max(x)-lim_b)/1000,(np.max(x)-lim_a)/1000])
    # plt.ylim([np.min(y),0.])
    # plt.colorbar(pGlass,orientation='horizontal', fraction=0.046)
    # plt.title('RIGHT OUTCROP')



    # # u velocity plot
    # varMat = u*(3.14e7)#ca0
    # varStep = u*(3.14e7)#ca
    # contours = np.linspace(np.min(varMat),np.max(varMat),10)
    #
    # ax1=fig.add_subplot(2,2,1, aspect=asp/4.0,frameon=False)
    # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    # plt.xticks(domain_x_ticks,domain_x_tick_labels)
    # plt.yticks(domain_y_ticks)
    # for c in pGlass.collections:
    #     c.set_edgecolor("face")
    #
    # cbar= plt.colorbar(pGlass, orientation='horizontal')
    # cbar.set_ticks(np.linspace(np.min(varStep),np.max(varStep),num=bar_bins,endpoint=True))
    # cbar.ax.set_xlabel('u [m/yr]')
    # cbar.solids.set_edgecolor("face")
    #
    #
    # # v velocity plot
    # varMat = v*(3.14e7)#dic0
    # varStep = v*(3.14e7)#dic
    # contours = np.linspace(np.min(varMat),np.max(varMat),10)
    #
    # ax1=fig.add_subplot(2,2,2, aspect=asp/4.0,frameon=False)
    # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,antialiased=True)
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    # plt.xticks(domain_x_ticks,domain_x_tick_labels)
    # plt.yticks(domain_y_ticks)
    # for c in pGlass.collections:
    #     c.set_edgecolor("face")
    #
    # cbar= plt.colorbar(pGlass, orientation='horizontal')
    # cbar.set_ticks(np.linspace(np.min(varStep),np.max(varStep),num=bar_bins,endpoint=True))
    # cbar.ax.set_xlabel('v [m/yr]')
    # cbar.solids.set_edgecolor("face")

    plt.savefig(outpath+'a_paper_'+str(i+restart)+'.png',bbox_inches='tight')
    # plt.savefig(outpath+'b_paper_'+str(i+restart)+'.eps',bbox_inches='tight')













    # #todo: FIG: jdfaq.png
    #
    # #-aquifer
    #
    # fig=plt.figure()
    #
    #
    # # aqx = 17
    # # aqx2 = (len(x)) - 17
    # aqx = int((param_w/50.0)) +30 #+ 20
    # aqx2 = len(x) - int((param_w_rhs/50.0)) - 32 #- 40
    # aqy = 0
    # aqy2 = len(y)
    #
    #
    # # u velocity in the channel
    # varMat = u[aqy:aqy2,aqx:aqx2]*(3.14e7)#ca0
    # varStep = u[aqy:aqy2,aqx:aqx2]*(3.14e7)#ca
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    # scanned = varStep[:,bitsx/2]
    # #print scanned
    # print "mean scanned qu" , np.mean(scanned[np.abs(scanned)>0.001])
    # # print "sum scanned qu" , np.sum(scanned[scanned>0.001])/(200.0/(y[1]-y[0]))
    # print "sum scanned qu" , np.sum(scanned[scanned>0.001])/((200.0+1.5*(y[1]-y[0]))/(y[1]-y[0]))
    # print "mmaaxx scanned qu" , np.max(scanned)
    # # print "max qu" , np.max(scanned)
    # # print "mean psi" , np.mean(psi)
    # print "max psi" , np.max(psi)
    # # print "max psi, aquifer" , np.max(psi[aqy:aqy2,aqx:aqx2])
    # # print "mean psi, aquifer" , np.mean(psi[aqy:aqy2,aqx:aqx2])
    #
    # #print np.sum(heapq.nlargest(4,scanned))/4.0
    #
    # print " "
    #
    # if i >= 4:
    #     conv_count = conv_count + 1
    #     # conv_mean_qu = conv_mean_qu + np.sum(scanned[scanned>0.001])/(200.0/(y[1]-y[0]))
    #     conv_mean_qu = conv_mean_qu + np.mean(scanned[np.abs(scanned)>0.1])#np.sum(scanned)/(200.0/(y[1]-y[0]))
    #     conv_max_qu = conv_max_qu + np.max(varStep)
    #     conv_mean_psi = conv_mean_psi + np.mean(psi)
    #     conv_max_psi = conv_max_psi + np.max(psi)
    # if i == 9:
    # #if i >= 4:
    #     print "means over time:"
    #     print "conv_mean_qu" , conv_mean_qu/float(conv_count)
    #     print "conv_max_qu" , conv_max_qu/float(conv_count)
    #     print "conv_mean_psi" , conv_mean_psi/float(conv_count)
    #     print "conv_max_psi" , conv_max_psi/float(conv_count)
    #
    # ax1=fig.add_subplot(2,1,1, aspect=asp/4.0,frameon=False)
    # pGlass=plt.contourf(x[aqx:aqx2],y[aqy:aqy2],varStep,contours,cmap=cm.rainbow,alpha=1.0,antialiased=True)
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    # p = plt.contour(xg[aqy:aqy2,aqx:aqx2],yg[aqy:aqy2,aqx:aqx2],perm[aqy:aqy2,aqx:aqx2],[-14.9],colors='black',linewidths=np.array([0.5]))
    # plt.xticks(domain_x_ticks,domain_x_tick_labels)
    # for c in pGlass.collections:
    #     c.set_edgecolor("face")
    #
    # plt.ylim([y[aqy],y[aqy2-1]])
    # cbar= plt.colorbar(pGlass, orientation='horizontal')
    # cbar.set_ticks(np.linspace(np.min(varStep),np.max(varStep),num=bar_bins,endpoint=True))
    # cbar.ax.set_xlabel('u [m/yr]')
    # cbar.solids.set_edgecolor("face")
    #
    #
    #
    # varMat = v[aqy:aqy2,aqx:aqx2]*(3.14e7)#c14[aqy:aqy2,aqx:aqx2]#*(3.14e7)#dic0
    # varStep = v[aqy:aqy2,aqx:aqx2]*(3.14e7)#c14[aqy:aqy2,aqx:aqx2]#*(3.14e7)#dic
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    #
    # ax1=fig.add_subplot(2,1,2, aspect=asp/4.0,frameon=False)
    # pGlass = plt.contourf(x[aqx:aqx2], y[aqy:aqy2], varStep, contours, cmap=cm.rainbow, alpha=1.0,antialiased=True)
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
    # p = plt.contour(xg[aqy:aqy2,aqx:aqx2],yg[aqy:aqy2,aqx:aqx2],perm[aqy:aqy2,aqx:aqx2],[-14.9],colors='black',linewidths=np.array([0.5]))
    # plt.xticks(domain_x_ticks,domain_x_tick_labels)
    # for c in pGlass.collections:
    #     c.set_edgecolor("face")
    #
    #
    #
    # plt.ylim([y[aqy],y[aqy2-1]])
    # cbar= plt.colorbar(pGlass, orientation='horizontal')
    # cbar.set_ticks(np.linspace(np.min(varStep),np.max(varStep),num=bar_bins,endpoint=True))
    # cbar.ax.set_xlabel('v [m/yr]')
    # cbar.solids.set_edgecolor("face")
    #
    #
    # plt.savefig(outpath+'jdfaq_'+str(i+restart)+'.png',bbox_inches='tight')



    xn = len(x)
    lim_a = 0.0
    lim_b = 1000.0
    lim_a0 = int(lim_a/(x[1]-x[0]))
    lim_b0 = int(lim_b/(x[1]-x[0]))
    lim_u = 0
    lim_o = len(y)

    aspSQ = asp/80.0
    aspZ = asp

    # if i==0:

        #todo: FIG: zoom plot

        # fig=plt.figure()
        #
        # varMat = maskP[lim_u:lim_o,lim_a0:lim_b0]
        # varStep = maskP[lim_u:lim_o,lim_a0:lim_b0]
        # contours = np.linspace(np.min(varMat),np.max(varMat),10)
        #
        # ax1=fig.add_subplot(2,2,1,aspect=aspSQ/1000.0,frameon=False)
        # pGlass = plt.pcolor(x[lim_a0:lim_b0]/1000.0, y[lim_u:lim_o], varStep)
        # #p = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0],yg[lim_u:lim_o,lim_a0:lim_b0],perm[lim_u:lim_o,lim_a0:lim_b0],
        # #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        #
        # plt.xlim([lim_a/1000.0,lim_b/1000.0])
        # plt.title('LEFT OUTCROP maskP')
        #
        # varMat = maskP[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        # varStep = maskP[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        # contours = np.linspace(np.min(varMat),np.max(varMat),10)
        #
        # ax1=fig.add_subplot(2,2,2,aspect=aspSQ/1000.0,frameon=False)
        # pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0]/1000.0, y[lim_u:lim_o], varStep)
        # #p = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0],
        # #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        #
        # plt.xlim([(np.max(x)-lim_b)/1000.0,(np.max(x)-lim_a)/1000.0])
        # plt.title('RIGHT OUTCROP maskP')
        #
        #
        # varMat = perm#mask+maskP
        # varStep = perm#mask+maskP
        # contours = np.linspace(np.min(varMat),np.max(varMat),10)
        #
        # ax1=fig.add_subplot(2,1,2,aspect=asp/1000.0,frameon=False)
        # pGlass = plt.pcolor(x/1000.0, y, varStep)
        # cMask = plt.contour(xg/1000.0,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
        #
        # plt.title('LEFT OUTCROP mask')
        #
        #
        # plt.savefig(outpath+'jdfZoom_'+str(i)+'.png',bbox_inches='tight')


        # #todo: FIG: jdfZoomK_0.png
        #
        # fig=plt.figure(figsize=(6.0,8.0))
        #
        # varMat = perm[lim_u:lim_o,lim_a0:lim_b0]
        # varStep = perm[lim_u:lim_o,lim_a0:lim_b0]
        # contours = np.linspace(np.min(varMat),np.max(varMat),10)
        #
        # ax1=fig.add_subplot(3,2,1,aspect=aspSQ/1000.0,frameon=False)
        # pGlass = plt.pcolor(x[lim_a0:lim_b0]/1000.0, y[lim_u:lim_o], varStep)
        # #p = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0],yg[lim_u:lim_o,lim_a0:lim_b0],perm[lim_u:lim_o,lim_a0:lim_b0],
        # #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        #
        # plt.xlim([lim_a/1000.0,lim_b/1000.0])
        # plt.ylim([-1250.0, 0.0])
        # plt.title('LEFT OUTCROP kx')
        #
        # varMat = perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        # varStep = perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        # contours = np.linspace(np.min(varMat),np.max(varMat),10)
        #
        # ax1=fig.add_subplot(3,2,2,aspect=aspSQ/1000.0,frameon=False)
        # pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0]/1000.0, y[lim_u:lim_o], varStep)
        # #p = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0],
        # #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        #
        # plt.xlim([(np.max(x)-lim_b)/1000.0,(np.max(x)-lim_a)/1000.0])
        # plt.ylim([-1250.0, 0.0])
        # plt.title('RIGHT OUTCROP kx')
        #
        #
        #
        #
        # varMat = perm[lim_u:lim_o,lim_a0:lim_b0]
        # varStep = perm[lim_u:lim_o,lim_a0:lim_b0]
        # contours = np.linspace(np.min(varMat),np.max(varMat),10)
        #
        # ax1=fig.add_subplot(3,2,3,aspect=aspSQ/1000.0,frameon=False)
        # pGlass = plt.pcolor(x[lim_a0:lim_b0]/1000.0, y[lim_u:lim_o], varStep)
        # #p = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0],yg[lim_u:lim_o,lim_a0:lim_b0],perm[lim_u:lim_o,lim_a0:lim_b0],
        # #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        #
        # plt.xlim([lim_a/1000.0,lim_b/1000.0])
        # plt.ylim([-1250.0, 0.0])
        # plt.title('LEFT OUTCROP ky')
        #
        # varMat = perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        # varStep = perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        # contours = np.linspace(np.min(varMat),np.max(varMat),10)
        #
        # ax1=fig.add_subplot(3,2,4,aspect=aspSQ/1000.0,frameon=False)
        # pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0]/1000.0, y[lim_u:lim_o], varStep)
        # #p = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0],
        # #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        #
        # plt.xlim([(np.max(x)-lim_b)/1000.0,(np.max(x)-lim_a)/1000.0])
        # plt.ylim([-1250.0, 0.0])
        # plt.title('RIGHT OUTCROP ky')
        #
        #
        #
        #
        #
        #
        #
        # varMat = maskP[lim_u:lim_o,lim_a0:lim_b0]
        # varStep = maskP[lim_u:lim_o,lim_a0:lim_b0]
        # contours = np.linspace(np.min(varMat),np.max(varMat),10)
        #
        # ax1=fig.add_subplot(3,2,5,aspect=aspSQ/1000.0,frameon=False)
        # pGlass = plt.pcolor(x[lim_a0:lim_b0]/1000.0, y[lim_u:lim_o], varStep)
        # #p = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0],yg[lim_u:lim_o,lim_a0:lim_b0],perm[lim_u:lim_o,lim_a0:lim_b0],
        # #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        #
        # plt.xlim([lim_a/1000.0,lim_b/1000.0])
        # plt.ylim([-1250.0, 0.0])
        # plt.title('LEFT OUTCROP maskP')
        #
        # varMat = maskP[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        # varStep = maskP[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
        # contours = np.linspace(np.min(varMat),np.max(varMat),10)
        #
        # ax1=fig.add_subplot(3,2,6,aspect=aspSQ/1000.0,frameon=False)
        # pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0]/1000.0, y[lim_u:lim_o], varStep)
        # #p = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0],perm[lim_u:lim_o,xn-lim_b0:xn-lim_a0],
        # #[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
        #
        # plt.xlim([(np.max(x)-lim_b)/1000.0,(np.max(x)-lim_a)/1000.0])
        # plt.ylim([-1250.0, 0.0])
        # plt.title('RIGHT OUTCROP maskP')
        #
        #
        #
        # plt.savefig(outpath+'jdfZoomK_'+str(i)+'.png',bbox_inches='tight')


    # #todo: FIG: zoomVel_
    # temp_max = 180.0
    #
    # fig=plt.figure(figsize=(9.0,9.0))
    #
    # varMat = temp[lim_u:lim_o,lim_a0:lim_b0]
    # varStep = temp[lim_u:lim_o,lim_a0:lim_b0]
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    #
    # ax1=fig.add_subplot(2,2,1, aspect=aspSQ/1000,frameon=False)
    # pGlass = plt.contourf(x[lim_a0:lim_b0]/1000, y[lim_u:lim_o], varStep, 40, cmap=cm.rainbow,vmin = np.min(temp),vmax=180)
    # CS = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0]/1000, yg[lim_u:lim_o,lim_a0:lim_b0], psi[lim_u:lim_o,lim_a0:lim_b0], 8, colors='black',linewidths=np.array([0.5]))
    # p = plt.contour(xg/1000,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    # cMask = plt.contour(xg/1000,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    #
    # plt.xlim([lim_a/1000.0,lim_b/1000.0])
    # plt.ylim([np.min(y),0.])
    # cbar = plt.colorbar(pGlass,orientation='horizontal', fraction=0.046)
    # # cbar.set_ticks(np.linspace(np.min(temp),temp_max,5))
    # # cbar.set_clim(np.min(temp), temp_max)
    # plt.title('LEFT OUTCROP')
    #
    #
    # varMat = temp[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
    # varStep = temp[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    #
    # ax1=fig.add_subplot(2,2,2, aspect=aspSQ/1000,frameon=False)
    # pGlass = plt.contourf(x[xn-lim_b0:xn-lim_a0]/1000, y[lim_u:lim_o], varStep, cmap=cm.rainbow,vmin = np.min(temp),vmax=180)
    # CS = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0]/1000, yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0], psi[lim_u:lim_o,xn-lim_b0:xn-lim_a0], 8, colors='black',linewidths=np.array([0.5]))
    # p = plt.contour(xg/1000,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    # cMask = plt.contour(xg/1000,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    #
    # plt.xlim([(np.max(x)-lim_b)/1000,(np.max(x)-lim_a)/1000])
    # plt.ylim([np.min(y),0.])
    # plt.colorbar(pGlass,orientation='horizontal', fraction=0.046)
    # plt.title('RIGHT OUTCROP')
    #
    #
    # varStep = psi[lim_u:lim_o,lim_a0:lim_b0]
    # varMat = varStep
    #
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    # ax1=fig.add_subplot(2,2,3, aspect=aspSQ/1000,frameon=False)
    # pGlass = plt.pcolor(x[lim_a0:lim_b0]/1000, y[lim_u:lim_o], varStep, cmap=cm.rainbow)
    # #p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    # #cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    #
    # plt.xlim([lim_a/1000,lim_b/1000])
    # plt.ylim([np.min(y),0.])
    #
    #
    # varStep = psi[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
    # varMat = varStep
    #
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    # ax1=fig.add_subplot(2,2,4, aspect=aspSQ/1000,frameon=False)
    # pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0]/1000, y[lim_u:lim_o], varStep, cmap=cm.rainbow)
    # #p = plt.contour(xg,yg,perm,[-15.9],colors='black',linewidths=np.array([1.5]))
    # #cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    #
    # plt.xlim([(np.max(x)-lim_b)/1000,(np.max(x)-lim_a)/1000])
    # plt.ylim([np.min(y),0.])
    #
    # plt.savefig(outpath+'jdfZoomVel_'+str(i+restart)+'.png',bbox_inches='tight')






    # #todo: FIG: zoom_u_v
    #
    # fig=plt.figure()
    #
    # varMat = u[:,lim_a0:lim_b0]*(3.14e7)
    # varStep = u[:,lim_a0:lim_b0]*(3.14e7)
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    #
    # ax1=fig.add_subplot(2,2,1, aspect=aspSQ,frameon=False)
    # pGlass = plt.pcolor(x[lim_a0:lim_b0], y, varStep, cmap=cm.rainbow)
    # p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    #
    # plt.colorbar(pGlass,orientation='horizontal',label='x velocity [m/yr]')
    #
    # plt.xlim([lim_a,lim_b])
    # plt.ylim([np.min(y),0.])
    # plt.title('LEFT OUTCROP')
    #
    #
    # varMat = u[:,xn-lim_b0:xn-lim_a0]*(3.14e7)
    # varStep = u[:,xn-lim_b0:xn-lim_a0]*(3.14e7)
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    #
    # ax1=fig.add_subplot(2,2,2, aspect=aspSQ,frameon=False)
    # pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y, varStep,cmap=cm.rainbow)
    # p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    #
    # plt.colorbar(pGlass,orientation='horizontal',label='x velocity [m/yr]')
    #
    # plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
    # plt.ylim([np.min(y),0.])
    # plt.title('RIGHT OUTCROP')
    #
    #
    #
    # varMat = v[:,lim_a0:lim_b0]*(3.14e7)
    # varStep = v[:,lim_a0:lim_b0]*(3.14e7)
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    #
    # ax1=fig.add_subplot(2,2,3, aspect=aspSQ,frameon=False)
    # pGlass = plt.pcolor(x[lim_a0:lim_b0], y, varStep, cmap=cm.rainbow)
    # p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    #
    # plt.colorbar(pGlass,orientation='horizontal',label='y velocity [m/yr]')
    #
    # plt.xlim([lim_a,lim_b])
    # plt.ylim([np.min(y),0.])
    #
    #
    # varMat = v[:,xn-lim_b0:xn-lim_a0]*(3.14e7)
    # varStep = v[:,xn-lim_b0:xn-lim_a0]*(3.14e7)
    # contours = np.linspace(np.min(varMat),np.max(varMat),20)
    #
    # ax1=fig.add_subplot(2,2,4, aspect=aspSQ,frameon=False)
    # pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y, varStep,cmap=cm.rainbow)
    # p = plt.contour(xg,yg,perm,[-14.9],colors='black',linewidths=np.array([1.5]))
    # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))
    #
    # plt.colorbar(pGlass,orientation='horizontal',label='y velocity [m/yr]')
    #
    # plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
    # plt.ylim([np.min(y),0.])
    #
    #
    #
    # plt.savefig(outpath+'jdfZoom_u_v_'+str(i+restart)+'.png')




#todo: jdf_u_ts.png
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)

plt.plot(u_ts)

#plt.ylim([-0.2,0.2])

plt.savefig(outpath+'jdf_u_ts.png',bbox_inches='tight')
