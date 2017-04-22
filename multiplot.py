# revived_JDF.py

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




def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
    
    
    
    

##############
# INITIALIZE #
##############

cell = 10
#steps = 400
steps = 5
minNum = 58

tsw = 3040
path = "output/revival/chem_up_1207/"
path_ex = "output/revival/chem_up_1207/tsw" + str(tsw) + "/"

# load output
x = np.loadtxt(path_ex + 'x.txt',delimiter='\n')
y = np.loadtxt(path_ex + 'y.txt',delimiter='\n')
asp = np.abs(np.max(x)/np.min(y))/4.0
xg, yg = np.meshgrid(x[:],y[:])

bitsx = len(x)
bitsy = len(y)

xCell = x
yCell = y
xCell = xCell[::cell]
yCell= yCell[::cell]
xCell = np.append(xCell, np.max(xCell)+xCell[1])
yCell = np.append(yCell, np.max(yCell)-yCell[-1])
bitsCx = len(xCell)
bitsCy = len(yCell)

ripTrans = np.transpose(mpd.ripSort)
hf_interp = np.zeros(160)
hf_interp = np.interp(x,ripTrans[0,:],ripTrans[1,:])


xg, yg = np.meshgrid(x[:],y[:])

#param_h = np.array([150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
param_h = np.array([200.0])
param_w = np.array([1000.0])
param_o = np.array([300.0])
# param_orhs = np.array([300.0, 400.0, 500.0])
param_orhs = np.array([500.0])

u_mean = np.zeros((len(param_h)+1,len(param_w)+1,len(param_o)+1,len(param_orhs)+1))
u_max = np.zeros((len(param_h)+1,len(param_w)+1,len(param_o)+1,len(param_orhs)+1))
u_middle = np.zeros((len(param_h)+1,len(param_w)+1,len(param_o)+1,len(param_orhs)+1))
hf_range = np.zeros((len(param_h)+1,len(param_w)+1,len(param_o)+1,len(param_orhs)+1))

for i in range(len(param_h)):
    for j in range(len(param_w)):
        for k in range(len(param_o)):
            for l in range(len(param_orhs)):
            
                sim_name = "o" + str(int(param_o[k])) + "w" + str(int(param_w[j])) + "h" + str(int(param_h[i]))  + "orhs" + str(int(param_orhs[l])) + ""
                path_sim = path + "o" + str(int(param_o[k])) + "w" + str(int(param_w[j])) + "h" + str(int(param_h[i]))  + "orhs" + str(int(param_orhs[l])) + "/"
                #path_sim = path + "o" + str(int(param_o[k])) + "w" + str(int(1000.0)) + "h" + str(int(param_h[i])) + "/transfer/"
                # JUST FOR CHEM ICS
                path_sim = path_ex
                sim_name = str(tsw)
                
                print ""
                print sim_name
            
                # load stuff
                uMat0 = np.loadtxt(path_sim + 'uMat.txt')*(3.14e7)#*10.0
                vMat0 = np.loadtxt(path_sim + 'vMat.txt')*(3.14e7)#*10.0
                psiMat0 = np.loadtxt(path_sim + 'psiMat.txt')
                hMat0 = np.loadtxt(path_sim + 'hMat.txt')
                permMat0 = np.log10(np.loadtxt(path_sim + 'permMat.txt'))
                mask = np.loadtxt(path_sim + 'mask.txt')
                lambdaMat = np.loadtxt(path_sim + 'lambdaMat.txt')
                step = 5
                u_last = uMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
                v_last = vMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
                psi_last = psiMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
                h_last = hMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
                perm_last = permMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
            
                cap = 20
                # capy = (np.max(param_o[k],param_orhs[l])/(y[1]-y[0])) - 5
                if param_orhs[l] > param_o[k]:
                    capy = ((param_orhs[l])/(y[1]-y[0])) + 5
                if param_orhs[l] <= param_o[k]:
                    capy = ((param_o[k])/(y[1]-y[0])) + 5
                
                # mean u
                colMax = np.zeros(len(x))
                for n in range(len(x[cap:-cap])):
                    if sum(u_last[cap:-cap,len(x)/2]) > 0.0:
                        colMax[n] = np.max(u_last[capy:-capy,n])
                    if sum(u_last[cap:-cap,len(x)/2]) < 0.0:
                        colMax[n] = np.min(u_last[capy:-capy,n])
                colMean = np.sum(colMax)/len(x[cap:-cap])
                print "mean u"
                print colMean
                u_mean[i,j,k,l] = colMean
            
                # max u
                print "max u"
                if (u_mean[i,j,k,l] > 0.0):
                    u_max[i,j,k,l] = np.max(u_last)
                if (u_mean[i,j,k,l] < 0.0):
                    u_max[i,j,k,l] = np.min(u_last)

            
            
                ##############################
                ##    PLOT LAST TIMESTEP    ##
                ##############################
                fig=plt.figure()
            
                varMat = hMat0
                varStep = h_last

                contours = np.linspace(np.min(varStep),np.max(varMat),12)
                ax1=fig.add_subplot(2,1,2, aspect=asp,frameon=False)
                pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
                cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
                cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
                p = plt.contour(xg,yg,perm_last,[-13.15,-17.25],colors='black',linewidths=np.array([2.0]))
                CS = plt.contour(xg, yg, psi_last, 10, colors='black',linewidths=np.array([0.5]))
                cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)

                varMat = vMat0
                varStep = v_last

                contours = np.linspace(np.min(varMat),np.max(varMat),10)
                ax1=fig.add_subplot(2,2,2, aspect=asp,frameon=False)
                pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
                cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
                cbar.ax.set_xlabel('v [m/yr]')
                p = plt.contour(xg,yg,perm_last,[-12.15,-17.25],colors='black',linewidths=np.array([1.0]))
                cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
    

                varMat = uMat0
                varStep = u_last
    
                contours = np.linspace(np.min(varMat),np.max(varMat),10)
                ax1=fig.add_subplot(2,2,1, aspect=asp,frameon=False)
                pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
                cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
                cbar.ax.set_xlabel('u [m/yr]')
                p = plt.contour(xg,yg,perm_last,[-12.15,-17.25],colors='black',linewidths=np.array([1.0]))
                cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
    
                plt.savefig(path+sim_name+'_'+str(step)+'.png')
                
                
                ################
                # AQUIFER PLOT #
                ################

                fig=plt.figure()
    
                varMat = v_last[capy:-capy,cap:-cap]
                varStep = v_last[capy:-capy,cap:-cap]

                contours = np.linspace(np.min(varMat),np.max(varMat),10)
                ax1=fig.add_subplot(2,1,2, aspect=asp,frameon=False)
                pGlass = plt.contourf(x[cap:-cap], y[capy:-capy], varStep, contours,
                                     cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
                cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
                cbar.ax.set_xlabel('v [m/yr]')
                p = plt.contour(xg,yg,perm_last,[-12.15,-17.25],colors='black',linewidths=np.array([1.0]))
                cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
                #plt.xlim([np.max(x)/5.0,np.max(x)*4.0/5.0])
    

                varMat = u_last[capy:-capy,cap:-cap]
                varStep = u_last[capy:-capy,cap:-cap]
    
                contours = np.linspace(np.min(varMat),np.max(varMat),10)
                ax1=fig.add_subplot(2,1,1, aspect=asp,frameon=False)
                pGlass = plt.contourf(x[cap:-cap], y[capy:-capy], varStep, contours,
                                     cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
                cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
                cbar.ax.set_xlabel('u [m/yr]')
                p = plt.contour(xg,yg,perm_last,[-12.15,-17.25],colors='black',linewidths=np.array([1.0]))
                cMask = plt.contourf(xg,yg,mask,[0.0,0.5],colors='white',alpha=1.0,zorder=10)
                #plt.xlim([np.max(x)/5.0,np.max(x)*4.0/5.0])


    
    
                plt.savefig(path+'aq_'+sim_name+'.png')
                
            
            
                ##########################
                ##    PLOT HEAT FLOW    ##
                ##########################
            
                fig=plt.figure()
                ax1=fig.add_subplot(1,1,1)
    
                heatVec = np.zeros(len(h_last[-1,:]))
                heatSed = np.zeros(len(h_last[-1,:]))
                heatBottom = np.zeros(len(h_last[-1,:]))
                heatLeft = np.zeros(len(h_last[:,-1]))
                heatRight = np.zeros(len(h_last[:,-1]))
                for m in range(len(x)):
                    heatBottom[m] = -1000.0*2.0*(h_last[2,m] - h_last[1,m])/(y[1]-y[0])
                    for n in range(len(y)):
                        if (mask[n,m] == 25.0) or (mask[n,m] == 50.0):
                            heatVec[m] = -1000.0*lambdaMat[n-1,m]*(h_last[n+1,m] - h_last[n,m])/(y[1]-y[0])
                            heatSed[m] = -1000.0*lambdaMat[n-1,m]*(h_last[n+1,m] - h_last[n,m])/(y[1]-y[0])
                    for n in range(len(y)):
                        if mask[n,m] == 5.0:
                            heatLeft[n] = -1000.0*lambdaMat[n,m+1]*(h_last[n,m+1] - h_last[n,m])/(x[1]-x[0])
                           # heatVec[m-1] =  heatVec[m-1] + heatLeft[n]
                        if mask[n,m] == 10.0:
                            heatRight[n] = -1000.0*lambdaMat[n,m-1]*(h_last[n,m-1] - h_last[n,m])/(x[1]-x[0])
                            #heatVec[m+1] =  heatVec[m+1] + heatRight[n]

                #hf = -1000.0*lambdaMat[-1,:]*(h_last[-1,:] - h_last[-2,:])/(y[1]-y[0])

                
                #hf_range[i,j,k,l] = np.sum(abs((hf_interp[5:-5]-heatVec[5:-5])/hf_interp[5:-5]))/len(x)
                hf_range[i,j,k,l] = np.sum(abs((heatBottom[cap:-cap]-heatVec[cap:-cap])/heatBottom[cap:-cap]))/len(x)
                print hf_range[i,j,k,l]
    
                plt.ylim([-100.0,600.0])
                plt.xlim([np.min(x),np.max(x)])
                # ax1.plot(mpd.steinX1[::2],mpd.steinY1[::2], 'r-', label='Stein 2004 7.9 m/yr')
#                 ax1.plot(mpd.steinX2[::2],mpd.steinY2[::2], 'b-', label='Stein 2004 4.9 m/yr')
#                 p = plt.plot(x,hf_interp,'m',linewidth=1.0,label='HF interpolation')
                p = plt.plot(x,heatVec,'g',linewidth=2.0,label='Navah free flow outcrop simulation')
                p = plt.plot(x,heatSed,'b',linewidth=2.0,label='sediment only')
                p = plt.plot(x,heatBottom,'gold',linewidth=2.0,label='Conduction')
                #ax1.set_xticks(x,minor=True)
                #ax1.set_xticks(x[::5])
                #ax1.grid(which='minor', alpha=0.5)
                ax1.grid(which='major', alpha=1.0)

                #plt.scatter(mpd.ripXdata,mpd.ripQdata,10,color='k',label='ODP observations (Alt et al. 1996)',zorder=4)
    
                plt.ylabel('heat flow [mW/m^2]')
                plt.xlabel('x direction [m]')
                plt.legend(fontsize=8)

                plt.savefig(path+'ehf_'+sim_name+'.png')
            
#                 ##########################
#                 ##    PLOT SOMETHING ELSE    ##
#                 ##########################
#
#                 fig=plt.figure()
#                 ax1=fig.add_subplot(1,1,1)
#
#
#                 plt.ylim([-3.0,3.0])
#                 plt.xlim([-328.0,np.max(x)])
#                 # ax1.plot(mpd.steinX1[::2],mpd.steinY1[::2], 'r-', label='Stein 2004 7.9 m/yr')
# #                 ax1.plot(mpd.steinX2[::2],mpd.steinY2[::2], 'b-', label='Stein 2004 4.9 m/yr')
# #                 p = plt.plot(x,hf_interp,'m',linewidth=1.0,label='HF interpolation')
#                 p = plt.plot(x,colMax,'g',linewidth=2.0,label='Navah free flow outcrop simulation')
#                 #ax1.set_xticks(x,minor=True)
#                 #ax1.set_xticks(x[::5])
#                 #ax1.grid(which='minor', alpha=0.5)
#                 ax1.grid(which='major', alpha=1.0)
#
#                 #plt.scatter(mpd.ripXdata,mpd.ripQdata,10,color='k',label='ODP observations (Alt et al. 1996)',zorder=4)
#
#                 plt.ylabel('heat flow [mW/m^2]')
#                 plt.xlabel('x direction [m]')
#                 plt.legend(fontsize=8)
#
#                 plt.savefig(path+'evel_'+sim_name+'.eps')
#
            
            
          


######################
##    FIRST PLOT    ##
######################

fig=plt.figure()

x_param = param_orhs
y_param = param_h

x_shift = x_param[1]-x_param[0]
y_shift = y_param[1]-y_param[0]
print 'umean'
#u_mean[:,:,:,2] = 0.0
print u_mean[:,0,0,:]

x_param = np.append(x_param, x_param[-1]+x_shift)
y_param = np.append(y_param, y_param[-1]+y_shift)
print x_param.shape
print y_param.shape
u_mean2d = u_mean[:,0,0,:]
# u_mean2d[0,0] = 0.0
# u_mean2d[1,0] = 0.0
# u_mean2d[0,1] = 0.0
print u_mean2d

asp_multi = np.abs((np.max(x_param)-np.min(x_param))/(np.max(y_param)-np.min(y_param)))
col_min = -7.0
col_max = 7.0

mp = max(abs(np.min(u_mean2d))/(abs(np.max(u_mean2d))+abs(np.min(u_mean2d))),abs(np.max(u_mean2d))/(abs(np.max(u_mean2d))+abs(np.min(u_mean2d))))
print mp
orig_cmap = cm.bwr_r
shifted_cmap = shiftedColorMap(orig_cmap, midpoint=mp, name='shifted')

ax1=fig.add_subplot(1,2,1, aspect=asp_multi)

u_mean_pos = masked_array(u_mean,u_mean>0)
u_mean_neg = masked_array(u_mean,u_mean<0)
# pCol_pos = plt.pcolor(x_param, y_param, u_mean_pos[:,0,0,:],cmap=cm.Reds_r)
# pCol_neg = plt.pcolor(x_param, y_param, u_mean_neg[:,0,0,:],cmap=cm.Blues)
pCol = plt.pcolor(x_param, y_param, u_mean2d,cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('outflow outcrop height [m]', fontsize=10)
plt.ylabel('flow layer thickness [m]', fontsize=10)
plt.title('mean lateral flow velocity [m/yr]', fontsize=10)
# cax = fig.add_axes([0.44, 0.53, 0.02, 0.2])
# fig.colorbar(pCol_pos,cax=cax)
# cax = fig.add_axes([0.44, 0.74, 0.02, 0.2])
# fig.colorbar(pCol_neg,cax=cax)
#cax = fig.add_axes([0.44, 0.53, 0.02, 0.41])
#fig.colorbar(pCol, cax=cax, orientation='vertical')
plt.colorbar(pCol, orientation='horizontal')


ax1=fig.add_subplot(1,2,2, aspect=asp_multi)

print 'hfrange'
print hf_range[:,0,0,:]

pCol = plt.pcolor(x_param, y_param, hf_range[:,0,0,:]*100.0)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('outflow outcrop height [m]', fontsize=10)
plt.ylabel('flow layer thickness [m]', fontsize=10)
plt.title('total heat %% dissipated by fluid flow', fontsize=10)
# cax = fig.add_axes([0.915, 0.53, 0.02, 0.41])
# fig.colorbar(pCol,cax=cax)
plt.colorbar(pCol, orientation='horizontal')
# cax = fig.add_axes([0.9, 0.55, 0.03, 0.4])
# fig.colorbar(pCol, cax=cax, orientation='vertical')




fig.set_tight_layout(True)
plt.savefig(path+'pCol.png')





#
# ######################
# ##    SECOND PLOT    ##
# ######################
#
# fig=plt.figure()
#
# x_param = param_orhs
# y_param = param_h
#
# u_mean2d = u_mean[:-1,0,0,:-1]
# # u_mean2d = np.insert(u_mean2d, 0, u_mean2d[:,0], axis=1)
# # u_mean2d = np.insert(u_mean2d, 0, u_mean2d[0,:], axis=0)
#
#
#
# x_shift = x_param[1]-x_param[0]
# y_shift = y_param[1]-y_param[0]
# print 'umean'
# print u_mean2d
# print 'umax'
# print u_max[:,0,0,:]
#
#
# # x_param = np.insert(x_param, 0, x_param[0]-x_shift)
# # y_param = np.insert(y_param, 0, y_param[0]-y_shift)
# print x_param
# print y_param
# print u_mean[:,0,0,:].shape
# X, Y = np.meshgrid(x_param,y_param)
#
# print x_param.shape
# print y_param.shape
# print u_mean2d.shape
# print X.shape
# print Y.shape
#
# asp_multi = np.abs((np.max(x_param)-np.min(x_param))/(np.max(y_param)-np.min(y_param)))
#
# ax1=fig.add_subplot(2,2,1, aspect=asp_multi)
# pCol = plt.contourf(X, Y, u_mean2d, 15, cmap=shifted_cmap)
# plt.xticks(x_param)
# plt.yticks(y_param)
# plt.xlim([np.min(x_param), np.max(x_param)])
# plt.ylim([np.min(y_param), np.max(y_param)])
# plt.xlabel('outflow outcrop height [m]', fontsize=10)
# plt.ylabel('flow layer thickness [m]', fontsize=10)
# plt.title('mean lateral flow velocity [m/yr]', fontsize=10)
# cax = fig.add_axes([0.44, 0.53, 0.02, 0.41])
# fig.colorbar(pCol, cax=cax, orientation='vertical')
#
#
#
#
# ax1=fig.add_subplot(2,2,2, aspect=asp_multi)
#
#
# hf_2d = hf_range[:-1,0,0,:-1]
#
# pCol = plt.contourf(x_param, y_param, hf_2d*100.0, 15)
# plt.xticks(x_param)
# plt.yticks(y_param)
# plt.xlim([np.min(x_param), np.max(x_param)])
# plt.ylim([np.min(y_param), np.max(y_param)])
# plt.xlabel('outflow outcrop height [m]', fontsize=10)
# plt.ylabel('flow layer thickness [m]', fontsize=10)
# plt.title('total heat %% dissipated by fluid flow', fontsize=10)
# cax = fig.add_axes([0.915, 0.53, 0.02, 0.41])
# fig.colorbar(pCol,cax=cax, ticks=[30.0, 40.0, 50.0, 60.0, 70.0])
#
#
#
#
#
# fig.set_tight_layout(True)
# plt.savefig(path+'pCont.eps')




# #######################
# ##    SECOND PLOT    ##
# #######################
#
#
# fig=plt.figure()
#
# x_param = param_orhs
# y_param = param_h
#
# x_shift = x_param[1]-x_param[0]
# y_shift = y_param[1]-y_param[0]
# print 'umean'
# print u_mean[:,0,0,:]
# print 'umax'
# print u_max[:,0,0,:]
#
# x_param = np.append(x_param, x_param[-1]+x_shift)
# y_param = np.append(y_param, y_param[-1]+y_shift)
# print x_param.shape
# print y_param.shape
# print u_mean[:,0,0,:].shape
#
# asp_multi = np.abs((np.max(x_param)-np.min(x_param))/(np.max(y_param)-np.min(y_param)))
# col_min = np.min(u_mean)
# col_max = 3.0#np.max(u_max)
#
# ax1=fig.add_subplot(2,2,1, aspect=asp_multi)
#
#
# pCol = plt.contour(x_param, y_param, u_mean[:,0,0,:],5)
# plt.clabel(pCol, fontsize=9, inline=1)
# plt.xticks(x_param+x_shift/2.0, x_param)
# plt.yticks(y_param+y_shift/2.0, y_param)
# plt.xlim([np.min(x_param), np.max(x_param)])
# plt.ylim([np.min(y_param), np.max(y_param)])
# plt.xlabel('outflow outcrop height [m]', fontsize=10)
# plt.ylabel('flow layer thickness [m]', fontsize=10)
# plt.title('mean lateral flow velocity [m/yr]', fontsize=10)
#
#
# ax1=fig.add_subplot(2,2,2, aspect=asp_multi)
#
# pCol = plt.contour(x_param, y_param, u_max[:,0,0,:],5)
# plt.clabel(pCol, fontsize=9, inline=1)
# plt.xticks(x_param+x_shift/2.0, x_param)
# plt.yticks(y_param+y_shift/2.0, y_param)
# plt.xlim([np.min(x_param), np.max(x_param)])
# plt.ylim([np.min(y_param), np.max(y_param)])
# plt.xlabel('outflow outcrop height [m]', fontsize=10)
# plt.ylabel('flow layer thickness [m]', fontsize=10)
# plt.title('max lateral flow velocity [m/yr]', fontsize=10)
#
#
#
#
#
# fig.set_tight_layout(True)
# plt.savefig(path+'pCont.png')
#
#
#
