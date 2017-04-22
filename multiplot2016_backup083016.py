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
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
# plt.rc('title', fontsize=8)
# plt.rc('ylabel', labelsize=8)
# plt.rc('xlabel', labelsize=8)


plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"


sed = np.abs(mpd.interp_s)
sed1 = np.abs(mpd.interp_b)
sed2 = np.abs(mpd.interp_s - mpd.interp_b)

##############
# INITIALIZE #
##############

cell = 10
#steps = 400
steps = 5
minNum = 58

param_o_rhs = 600.0
param_o_rhs_string = '600'

tsw = 3040
# path = "output/revival/may16/31_h100_wrl500/"
# path_ex = "output/revival/may16/jdf_h100_orl200/w400wrhs450/"

# path = "output/revival/june16/crunch_klhs_g-1/"
# path_ex = "output/revival/june16/crunch_klhs_g-1/w200ch600/"

path = "output/revival/june16/crunch_noch1/"
path_ex = "output/revival/june16/crunch_noch1/w200klhs1e-12/"


# o300orhs500w1200wrhs1200h075

# load output
x = np.loadtxt(path_ex + 'x.txt',delimiter='\n')
y = np.loadtxt(path_ex + 'y.txt',delimiter='\n')
xn = len(x)
asp = np.abs(np.max(x)/np.min(y))/4.0
xg, yg = np.meshgrid(x[:],y[:])

bitsx = len(x)
bitsy = len(y)

ripTrans = np.transpose(mpd.ripSort)
hf_interp = np.zeros(200)
hf_interp = np.interp(x,ripTrans[0,:],ripTrans[1,:])


xg, yg = np.meshgrid(x[:],y[:])

# param_w = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0])
# param_w_string = ['400','450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950','1000']
# param_w_rhs = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0,1000.0])
# param_w_rhs_string = ['400','450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950','1000']


param_w = np.array([200.0, 400.0 ,600.0, 800.0, 1000.0, 1200.0, 1400.0])
param_w_string = ['200', '400', '600', '800', '1000', '1200', '1400']

# param_k_rhs = np.array([1, 2, 3])
# param_k_rhs_string = ['1e-13', '1e-14', '1e-15']

param_w_rhs = np.array([75.0, 100.0, 125.0, 150.0, 175.0, 200.0])
param_w_rhs_string = ['075', '100', '125', '150', '175', '200']

param_k_lhs = np.array([1, 2, 3, 4, 5, 6])
param_k_lhs_string = ['1e-12', '2e-12', '4e-12', '6e-12', '8e-12', '10e-12']

param_k_rhs = np.array([1, 2, 3, 4, 5, 6])
param_k_rhs_string = ['2e-12', '4e-12', '6e-12', '8e-12', '10e-12', '12e-12']

# param_k_rhs = np.array([1, 2, 3, 4, 5, 6, 7, 8 ])
# param_k_rhs_string = ['1e-12', '2e-12', '4e-12', '6e-12', '8e-12', '10e-12', '15e-12', '20e-12']

param_ch = np.array([600.0, 700.0, 800.0, 900.0])
param_ch_string = ['600', '700', '800', '900']

# param1 = param_o
# param1_string = param_o_string
# param2 = param_o_rhs
# param2_string = param_o_rhs_string

param1 = param_w
param1_string = param_w_string
param2 = param_k_lhs
param2_string = param_k_lhs_string

u_mean = np.zeros((len(param1)+1,len(param2)+1))
u_ts = np.zeros((len(param1)+1,len(param2)+1,5))
u_mean_ts = np.zeros((len(param1)+1,len(param2)+1))
u_max = np.zeros((len(param1)+1,len(param2)+1))
hf_range = np.zeros((len(param1)+1,len(param2)+1))
dom = np.zeros((len(param1)+1,len(param2)+1))




step1 = 1
step2 = 6

for i in range(len(param1)):
    for j in range(len(param2)):
        print " "
        print param1[i]
        print param2[j]
        
        #if param2[j] > param1[i]:
        if 2.0 > 1.0:
            u_last = np.zeros([len(y),len(x)])
            u_last_sum = np.zeros([len(y),len(x)])
            sim_name = 'w' + param1_string[i] + 'klhs' + param2_string[j]
            path_sim = path + sim_name + "/"

            print sim_name
    
            # load stuff
            uMat0 = np.loadtxt(path_sim + 'uMat.txt')*(3.14e7)#*10.0
            vMat0 = np.loadtxt(path_sim + 'vMat.txt')*(3.14e7)#*10.0
            psiMat0 = np.loadtxt(path_sim + 'psiMat.txt')
            hMat0 = np.loadtxt(path_sim + 'hMat.txt')
            #permMat0 = np.log10(np.loadtxt(path_sim + 'permMat.txt'))
            perm_last = np.log10(np.loadtxt(path_sim + 'permeability.txt'))
            maskP = np.loadtxt(path_sim + 'maskP.txt')
            lambdaMat = np.loadtxt(path_sim + 'lambdaMat.txt')
            step = 5
            v_last = vMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
            psi_last = psiMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
            h_last = hMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
            #perm_last = permMat0[:,(step-1.0)*len(x):(((step-1.0))*len(x)+len(x))]
            #u_last[perm_last < -13.0] = 0.0
        
            cap1 = int((param_w[i]/25.0))+30# + 12
            cap2 = int((100.0/25.0))+20# + 12
            # cap1 = int((500.0/50.0)) + 4
            # cap2 = int((500.0/50.0)) + 4
            print "cap1, cap2", cap1 , cap2
            cap = 19
            # capy = (np.max(param_o[k],param_orhs[l])/(y[1]-y[0])) - 5
            capy = 2#((param_o_rhs)/(y[1]-y[0])) + 5
            

            for stepk in range(step1,step2):
                u_last = uMat0[:,(stepk-1.0)*len(x):(((stepk-1.0))*len(x)+len(x))]
                u_last_sum = u_last_sum + uMat0[:,(stepk-1.0)*len(x):(((stepk-1.0))*len(x)+len(x))]

                u_max[i,j] = 0.0
                
                

                colMax = np.zeros(len(x))
                for n in range(cap1,len(x)-cap2):
                    cmax = np.max(u_last[capy:-capy,n])
                    cmin = np.min(u_last[capy:-capy,n])
                    if np.abs(cmax) > np.abs(cmin):
                        colMax[n] = cmax
                    if np.abs(cmax) < np.abs(cmin):
                        colMax[n] = cmin
                    #colMax[n] = cmax
                    if float(cmax) > .001 and float(cmin) <-.001:
                        colMax[n] = 0.0
                colMean = np.sum(colMax)/len(x[cap1:-cap2])
        
                if np.abs(colMax[cap1+1]) > np.abs(colMax[len(x)-cap2-1]):
                    dom[i,j] = 1
                if np.abs(colMax[cap1+1]) < np.abs(colMax[len(x)-cap2-1]):
                    dom[i,j] = 3
                m_vec = np.array([np.abs(colMax[cap1]),np.abs(colMax[len(x)-cap2])])
                #print m_vec
                # if colMax[cap1+5]/colMax[len(x)-cap2-5] < 0.0 and np.min(m_vec)/np.max(m_vec) > 0.1:
                #     dom[i,j] = 2
                # # if any(u_last[:,cap1]/u_last[:,cap2]) <= 0.0:
                # #     dom[i,j] = 2
                if np.max(np.abs(u_last[:,cap1]))/np.min(np.abs(u_last[:,cap2])) >= 0.01 and np.max(u_last[:,cap1])/np.min(u_last[:,cap2]) < -0.01:
                    dom[i,j] = 2
                if np.min(np.abs(u_last[:,cap1]))/np.max(np.abs(u_last[:,cap2])) >= 0.01 and np.min(u_last[:,cap1])/np.max(u_last[:,cap2]) < -0.01:
                    dom[i,j] = 2
                    
    

                u_mean[i,j] = colMean
                dmax = np.max(colMax)
                dmin = np.min(colMax)
                if np.abs(dmax) > np.abs(dmin):
                    u_max[i,j] = dmax
                if np.abs(dmax) < np.abs(dmin):
                    u_max[i,j] = dmin
                #u_max[i,j] = np.max(colMax)
                u_ts[i,j,stepk-1] = colMean
                # if colMean <= 0.0:
                #     u_ts[i,j,stepk-1] = 0.0
            
        
            
            u_mean_ts[i,j] = np.sum(u_ts[i,j,stepk-1])/1.0
            print "u_mean_ts[i,j]" , u_mean_ts[i,j]
            
            
            ##########################
            ##    PLOT U MEAN TS    ##
            ##########################
            fig=plt.figure()
            ax1=fig.add_subplot(1,1,1)

            plt.plot(u_ts[i,j,:])

            #plt.ylim([-0.2,0.2])

            plt.savefig(path+'u_mean_ts_'+sim_name+'_.png')

            #
            # ##############################
            # ##    PLOT LAST TIMESTEP    ##
            # ##############################
            # fig=plt.figure()
            #
            # varMat = hMat0
            # varStep = h_last[len(y)/2:,:]
            #
            # contours = np.linspace(np.min(varStep),np.max(varStep),12)
            # ax1=fig.add_subplot(1,1,1, aspect=asp,frameon=False)
            # pGlass = plt.contourf(x, y[len(y)/2:], varStep, contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
            # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
            # cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
            # p = plt.contour(xg,yg,perm_last,[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
            # CS = plt.contour(xg, yg, psi_last, 10, colors='black',linewidths=np.array([0.5]))
            # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
            # plt.ylim([y[0]/2.0,0.0])
            #
            # # varMat = vMat0
            # # varStep = v_last
            # #
            # # contours = np.linspace(np.min(varMat),np.max(varMat),10)
            # # ax1=fig.add_subplot(2,2,2, aspect=asp,frameon=False)
            # # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
            # # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
            # # cbar.ax.set_xlabel('v [m/yr]')
            # # p = plt.contour(xg,yg,perm_last,[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
            # # #CS = plt.contour(xg, yg, psi_last, 10, colors='black',linewidths=np.array([0.5]))
            # # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
            # #
            # #
            # # varMat = uMat0
            # # varStep = u_last
            # #
            # # contours = np.linspace(np.min(varMat),np.max(varMat),10)
            # # ax1=fig.add_subplot(2,2,1, aspect=asp,frameon=False)
            # # pGlass = plt.contourf(x, y, varStep, contours,cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
            # # cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
            # # cbar.ax.set_xlabel('u [m/yr]')
            # # p = plt.contour(xg,yg,perm_last,[-12.0,-13.5],colors='black',linewidths=np.array([2.0]))
            # # #CS = plt.contour(xg, yg, psi_last, 10, colors='black',linewidths=np.array([0.5]))
            # # cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
            #
            # plt.savefig(path+'jdf_'+sim_name+'_.png')
        
        
            ################
            # AQUIFER PLOT #
            ################
            
            

            fig=plt.figure()

            varMat = v_last[capy:-capy,cap1:-cap2]
            varStep = v_last[capy:-capy,cap1:-cap2]

            contours = np.linspace(np.min(varMat),np.max(varMat),10)
            ax1=fig.add_subplot(2,1,2, aspect=asp,frameon=False)
            pGlass = plt.contourf(x[cap1:-cap2], y[capy:-capy], varStep, contours,
                                 cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
            cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
            cbar.ax.set_xlabel('v [m/yr]')
            p = plt.contour(xg[capy:-capy,cap1:-cap2],yg[capy:-capy,cap1:-cap2],perm_last[capy:-capy,cap1:-cap2],
                                [-12.0,-13.5],colors='black',linewidths=np.array([1.0]))
            cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
            #plt.xlim([np.max(x)/5.0,np.max(x)*4.0/5.0])
            #plt.ylim([y[0],0.0])
            
            


            varMat = u_last[capy:-capy,cap1:-cap2]
            varStep = u_last[capy:-capy,cap1:-cap2]

            contours = np.linspace(np.min(varMat),np.max(varMat),10)
            ax1=fig.add_subplot(2,1,1, aspect=asp,frameon=False)
            pGlass = plt.contourf(x[cap1:-cap2], y[capy:-capy], varStep, contours,
                                 cmap=cm.rainbow, alpha=1.0,linewidth=0.0, color='#444444',antialiased=True)
            cbar= plt.colorbar(pGlass, orientation='horizontal',ticks=contours[::3])
            cbar.ax.set_xlabel('u [m/yr]')
            p = plt.contour(xg[capy:-capy,cap1:-cap2],yg[capy:-capy,cap1:-cap2],perm_last[capy:-capy,cap1:-cap2],
                                [-12.0,-13.5],colors='black',linewidths=np.array([1.0]))
            cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',linewidths=np.array([0.5]))
            #plt.xlim([np.max(x)/5.0,np.max(x)*4.0/5.0])
            #plt.ylim([y[0],0.0])



            plt.savefig(path+'aq_'+sim_name+'.eps')



            ######################################
            #        ZOOM OUTCROP PSI PLOT       #
            ######################################

            lim_a = 0.0
            lim_b = 2000.0
            lim_a0 = int(lim_a/(x[1]-x[0]))
            lim_b0 = int(lim_b/(x[1]-x[0]))
            lim_u = 0
            lim_o = len(y)

            fig=plt.figure()

            varMat = h_last[lim_u:lim_o,lim_a0:lim_b0]
            varStep = h_last[lim_u:lim_o,lim_a0:lim_b0]
            contours = np.linspace(np.min(varMat),np.max(varMat),20)

            ax1=fig.add_subplot(2,2,1, aspect=asp/4.0,frameon=False)
            pGlass = plt.contourf(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep, cmap=cm.rainbow)
            CS = plt.contour(xg[lim_u:lim_o,lim_a0:lim_b0], yg[lim_u:lim_o,lim_a0:lim_b0], psi_last[lim_u:lim_o,lim_a0:lim_b0], 8, colors='black',linewidths=np.array([0.5]))
            p = plt.contour(xg,yg,perm_last,[-15.9],colors='black',linewidths=np.array([1.5]))
            cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))

            plt.xlim([lim_a,lim_b])
            plt.ylim([np.min(y),0.])
            #plt.colorbar(orientation='horizontal')
            plt.title('LEFT OUTCROP')


            varMat = h_last[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
            varStep = h_last[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
            contours = np.linspace(np.min(varMat),np.max(varMat),20)

            ax1=fig.add_subplot(2,2,2, aspect=asp/4.0,frameon=False)
            pGlass = plt.contourf(x[xn-lim_b0:xn-lim_a0], y[lim_u:lim_o], varStep, cmap=cm.rainbow)
            CS = plt.contour(xg[lim_u:lim_o,xn-lim_b0:xn-lim_a0], yg[lim_u:lim_o,xn-lim_b0:xn-lim_a0], psi_last[lim_u:lim_o,xn-lim_b0:xn-lim_a0], 8, colors='black',linewidths=np.array([0.5]))
            p = plt.contour(xg,yg,perm_last,[-15.9],colors='black',linewidths=np.array([1.5]))
            cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))

            plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
            plt.ylim([np.min(y),0.])
            #plt.colorbar(orientation='horizontal')
            plt.title('RIGHT OUTCROP')


            varStep = psi_last[lim_u:lim_o,lim_a0:lim_b0]
            varMat = varStep

            contours = np.linspace(np.min(varMat),np.max(varMat),20)
            ax1=fig.add_subplot(2,2,3, aspect=asp/4.0,frameon=False)
            pGlass = plt.pcolor(x[lim_a0:lim_b0], y[lim_u:lim_o], varStep, cmap=cm.rainbow)
            p = plt.contour(xg,yg,perm_last,[-15.9],colors='black',linewidths=np.array([1.5]))
            cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))

            plt.xlim([lim_a,lim_b])
            plt.ylim([np.min(y),0.])


            varStep = psi_last[lim_u:lim_o,xn-lim_b0:xn-lim_a0]
            varMat = varStep

            contours = np.linspace(np.min(varMat),np.max(varMat),20)
            ax1=fig.add_subplot(2,2,4, aspect=asp/4.0,frameon=False)
            pGlass = plt.pcolor(x[xn-lim_b0:xn-lim_a0], y[lim_u:lim_o], varStep, cmap=cm.rainbow)
            p = plt.contour(xg,yg,perm_last,[-15.9],colors='black',linewidths=np.array([1.5]))
            cMask = plt.contour(xg,yg,maskP,[0.0,0.5],colors='w',alpha=1.0,linewidths=np.array([1.5]))

            plt.xlim([np.max(x)-lim_b,np.max(x)-lim_a])
            plt.ylim([np.min(y),0.])

            plt.savefig(path+'zoomVEL_'+sim_name+'.eps')



    

            # ##########################
            # ##    PLOT HEAT FLOW    ##
            # ##########################
            #
            # fig=plt.figure()
            # ax1=fig.add_subplot(1,1,1)
            #
            # heatVec = np.zeros(len(h_last[-1,:]))
            # heatSed = np.zeros(len(h_last[-1,:]))
            # heatBottom = np.zeros(len(h_last[-1,:]))
            # heatLeft = np.zeros(len(h_last[:,-1]))
            # heatRight = np.zeros(len(h_last[:,-1]))
            # for m in range(len(x)):
            #     heatBottom[m] = -1000.0*2.0*(h_last[2,m] - h_last[1,m])/(y[1]-y[0])
            #     for n in range(len(y)):
            #         if (maskP[n,m] == 25.0) or (maskP[n,m] == 50.0):
            #             heatVec[m] = -1000.0*lambdaMat[n-1,m]*(h_last[n+1,m] - h_last[n,m])/(y[1]-y[0])
            #             heatSed[m] = -1000.0*1.2*(h_last[n+1,m] - h_last[n,m])/(y[1]-y[0])
            #     for n in range(len(y)):
            #         if maskP[n,m] == 5.0:
            #             heatLeft[n] = -1000.0*lambdaMat[n,m+1]*(h_last[n,m+1] - h_last[n,m])/(x[1]-x[0])
            #            # heatVec[m-1] =  heatVec[m-1] + heatLeft[n]
            #         if maskP[n,m] == 10.0:
            #             heatRight[n] = -1000.0*lambdaMat[n,m-1]*(h_last[n,m-1] - h_last[n,m])/(x[1]-x[0])
            #             #heatVec[m+1] =  heatVec[m+1] + heatRight[n]
            #
            # #hf = -1000.0*lambdaMat[-1,:]*(h_last[-1,:] - h_last[-2,:])/(y[1]-y[0])
            #
            #
            #
            # heatVec = heatVec#*np.mean(sed1)/sed1#np.mean(sed2)/(sed2)
            # sh = cap1
            # shc = int(param_w[i]/130.0)
            #
            # plt.ylim([-100.0,800.0])
            # plt.xlim([np.min(x),np.max(x)])
            # plt.plot(mpd.steinX1[::2],mpd.steinY1[::2], 'k', linestyle="-.", lw=1, label='Stein 2004 7.9 m/yr')
            # plt.plot(mpd.steinX2[::2],mpd.steinY2[::2], 'k', linestyle="--", lw=1, label='Stein 2004 4.9 m/yr')
            # p = plt.plot(x,hf_interp,'k-',linewidth=1.0,label='HF interpolation')
            # p = plt.plot(x[sh-shc:-sh-shc],heatVec[sh:-sh]*np.min(sed1[sh:-sh]+sed[sh:-sh])/(sed1[sh:-sh]+sed[sh:-sh]),'r',linewidth=1.5,label='Model predicted heat flow')
            # p = plt.plot(x[sh-shc:-sh-shc],heatVec[sh:-sh]*np.min(sed1[sh:-sh])/(sed1[sh:-sh]),'g',linewidth=1.5,label='Model predicted heat flow')
            # p = plt.plot(x[sh:-sh],heatVec[sh:-sh],'b',linewidth=1.5,label='Model predicted heat flow')
            # p = plt.plot(x[sh:-sh],heatVec[sh:-sh]*np.mean(sed1[sh:-sh]+sed[sh:-sh])/(sed1[sh:-sh]+sed[sh:-sh]),'m',linewidth=1.5,label='Model predicted heat flow')
            # p = plt.plot(x[sh:-sh],heatVec[sh:-sh]*np.mean(sed1[sh:-sh])/(sed1[sh:-sh]),'c',linewidth=1.5,label='Model predicted heat flow')
            #
            # heat_actual = heatVec[sh:-sh]*np.min(sed1[sh:-sh])/(sed1[sh:-sh])
            #
            # plt.scatter(mpd.ripXdata,mpd.ripQdata,s=15,c='k',label='data')
            # #p = plt.plot(x,heatSed,'b',linewidth=2.0,label='sediment only')
            # p = plt.plot(x,heatBottom,'gold',linewidth=2.0,label='Conduction')
            # #ax1.set_xticks(x,minor=True)
            # #ax1.set_xticks(x[::5])
            # #ax1.grid(which='minor', alpha=0.5)
            # ax1.grid(which='major', alpha=1.0)
            #
            # print "heat flow error"
            # hf_range[i,j] =  np.sum(np.abs(hf_interp[sh-shc:-sh-shc] - heat_actual))/np.sum(np.abs(heat_actual))
            # print hf_range[i,j]
            # print " "
            #
            # #plt.scatter(mpd.ripXdata,mpd.ripQdata,10,color='k',label='ODP observations (Alt et al. 1996)',zorder=4)
            #
            # plt.ylabel('heat flow [mW/m^2]')
            # plt.xlabel('x direction [m]')
            # plt.legend(fontsize=8,loc='best',ncol=3)
            #
            # plt.savefig(path+'hf_'+sim_name+'.eps')
            #

            
            
          


######################
##    FIRST PLOT    ##
######################

fig=plt.figure()

x_param = param2
y_param = param1

x_shift = x_param[1]-x_param[0]
y_shift = y_param[1]-y_param[0]
print 'umean'
#u_mean[:,:,:,2] = 0.0
print u_mean_ts


x_param = np.append(x_param, x_param[-1]+x_shift)
y_param = np.append(y_param, y_param[-1]+y_shift)
print x_param.shape
print y_param.shape


asp_multi = np.abs((np.max(x_param)-np.min(x_param))/(np.max(y_param)-np.min(y_param)))
col_min = -7.0
col_max = 7.0



# ax1=fig.add_subplot(2,2,1, aspect=asp_multi)

print x_param.shape
print y_param.shape
print u_mean.shape

# u_mean_ts = np.insert(u_mean_ts, 0, u_mean_ts[:,0], axis=1)
# u_mean_ts = np.insert(u_mean_ts, 0, u_mean_ts[0,:], axis=0)
print u_mean_ts[-1,:].shape
print u_mean_ts[-2,:].shape
print "hi"
print u_mean_ts[-1,:]
print u_mean_ts[-2,:]
u_mean_ts[-1,:] = u_mean_ts[-2,:]
u_mean_ts[:,-1] = u_mean_ts[:,-2]

X, Y = np.meshgrid(x_param,y_param)


ax1=fig.add_subplot(1,2,1, aspect=asp_multi)

pCol = plt.pcolor(x_param, y_param, u_mean_ts, cmap=cm.jet)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('param w rhs [m]', fontsize=10)
plt.ylabel('param w [m]', fontsize=10)
# plt.xticks(param_w_rhs_string)
# plt.yticks(param_w_string)
plt.title('mean lateral flow velocity [m/yr]', fontsize=10)
plt.colorbar(pCol, orientation='horizontal')



ax1=fig.add_subplot(1,2,2, aspect=asp_multi)

print 'dominant outcrop'
print dom

pCol = plt.pcolor(x_param, y_param, dom)
plt.xticks(x_param+x_shift/2.0, x_param)
plt.yticks(y_param+y_shift/2.0, y_param)
plt.xlim([np.min(x_param), np.max(x_param)])
plt.ylim([np.min(y_param), np.max(y_param)])
plt.xlabel('param w rhs [m]', fontsize=10)
plt.ylabel('param w [m]', fontsize=10)
# plt.xticks(param_w_rhs_string)
# plt.yticks(param_w_string)
plt.title('left = 1, two flows = 2, right = 3', fontsize=10)

plt.colorbar(pCol, orientation='horizontal')



#fig.set_tight_layout(True)
plt.subplots_adjust(hspace=0.3)


st = fig.suptitle(path)
plt.savefig(path+'pCol.eps')



