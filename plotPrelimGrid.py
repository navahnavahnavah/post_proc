# plot prelimGrid.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


plt.rcParams['axes.color_cycle'] = "#FF3300, #FF9900, #FFCC00, \
                                    #00FF00, #339900, #009966, #0000FF, \
                                    #6600CC, #990099"


plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams['axes.linewidth'] = 2.0

temps = np.arange(2,42,2)


def grab(t):
    t_alk = np.zeros((len(temps)))
    t_dalk = np.zeros((len(temps)))
    t_calcite = np.zeros((len(temps)))
    t_dcalcite = np.zeros((len(temps)))
    t_ph = np.zeros((len(temps)))
    t_water = np.zeros((len(temps)))
    t_dglass = np.zeros((len(temps)))
    for i in range(len(temps)):
        bit = np.asarray(t[i])
        t_alk[i] = bit[0,3,-1]
        t_dalk[i] = bit[0,3,-1] - bit[0,3,-2]
        t_calcite[i] = bit[0,45,-1]
        t_dcalcite[i] = bit[0,63,-1] - bit[0,63,-2]
        t_ph[i] = bit[0,1,-1]
        t_water[i] = bit[0,83,-1]
        t_dglass[i] = bit[0,80,-1]
        

    output = [t_alk, t_calcite, t_dcalcite, t_ph, t_water, t_dalk, t_dglass]
    return output


    









#############################
# flush 1% / 10 years       #
# t = 3.14e8 (10 years)     #
#############################

t02 = np.loadtxt('f001_t02.txt')
t04 = np.loadtxt('f001_t04.txt')
t06 = np.loadtxt('f001_t06.txt')
t08 = np.loadtxt('f001_t08.txt')
t10 = np.loadtxt('f001_t10.txt')
t12 = np.loadtxt('f001_t12.txt')
t14 = np.loadtxt('f001_t14.txt')
t16 = np.loadtxt('f001_t16.txt')
t18 = np.loadtxt('f001_t18.txt')
t20 = np.loadtxt('f001_t20.txt')
t22 = np.loadtxt('f001_t22.txt')
t24 = np.loadtxt('f001_t24.txt')
t26 = np.loadtxt('f001_t26.txt')
t28 = np.loadtxt('f001_t28.txt')
t30 = np.loadtxt('f001_t30.txt')
t32 = np.loadtxt('f001_t32.txt')
t34 = np.loadtxt('f001_t34.txt')
t36 = np.loadtxt('f001_t36.txt')
t38 = np.loadtxt('f001_t38.txt')
t40 = np.loadtxt('f001_t40.txt')


t = [[t02[:,:]], [t04[:,:]], [t06[:,:]], [t08[:,:]], [t10[:,:]],
     [t12[:,:]], [t14[:,:]], [t16[:,:]], [t18[:,:]], [t20[:,:]],
     [t22[:,:]], [t24[:,:]], [t26[:,:]], [t28[:,:]], [t30[:,:]],
     [t32[:,:]], [t34[:,:]], [t36[:,:]], [t38[:,:]], [t40[:,:]]]


out = grab(t)

f001_dcalcite = out[2] #2.0*out[2] + out[5]*out[4]
f001_dglass = out[6]






#############################
# flush 2% / 10 years       #
# t = 3.14e8 (10 years)     #
#############################

t02 = np.loadtxt('f002_t02.txt')
t04 = np.loadtxt('f002_t04.txt')
t06 = np.loadtxt('f002_t06.txt')
t08 = np.loadtxt('f002_t08.txt')
t10 = np.loadtxt('f002_t10.txt')
t12 = np.loadtxt('f002_t12.txt')
t14 = np.loadtxt('f002_t14.txt')
t16 = np.loadtxt('f002_t16.txt')
t18 = np.loadtxt('f002_t18.txt')
t20 = np.loadtxt('f002_t20.txt')
t22 = np.loadtxt('f002_t22.txt')
t24 = np.loadtxt('f002_t24.txt')
t26 = np.loadtxt('f002_t26.txt')
t28 = np.loadtxt('f002_t28.txt')
t30 = np.loadtxt('f002_t30.txt')
t32 = np.loadtxt('f002_t32.txt')
t34 = np.loadtxt('f002_t34.txt')
t36 = np.loadtxt('f002_t36.txt')
t38 = np.loadtxt('f002_t38.txt')
t40 = np.loadtxt('f002_t40.txt')


t = [[t02[:,:]], [t04[:,:]], [t06[:,:]], [t08[:,:]], [t10[:,:]],
     [t12[:,:]], [t14[:,:]], [t16[:,:]], [t18[:,:]], [t20[:,:]],
     [t22[:,:]], [t24[:,:]], [t26[:,:]], [t28[:,:]], [t30[:,:]],
     [t32[:,:]], [t34[:,:]], [t36[:,:]], [t38[:,:]], [t40[:,:]]]


out = grab(t)

f002_dcalcite = out[2] #2.0*out[2] + out[5]*out[4]
f002_dglass = out[6]






#############################
# flush 3% / 10 years       #
# t = 3.14e8 (10 years)     #
#############################

t02 = np.loadtxt('f003_t02.txt')
t04 = np.loadtxt('f003_t04.txt')
t06 = np.loadtxt('f003_t06.txt')
t08 = np.loadtxt('f003_t08.txt')
t10 = np.loadtxt('f003_t10.txt')
t12 = np.loadtxt('f003_t12.txt')
t14 = np.loadtxt('f003_t14.txt')
t16 = np.loadtxt('f003_t16.txt')
t18 = np.loadtxt('f003_t18.txt')
t20 = np.loadtxt('f003_t20.txt')
t22 = np.loadtxt('f003_t22.txt')
t24 = np.loadtxt('f003_t24.txt')
t26 = np.loadtxt('f003_t26.txt')
t28 = np.loadtxt('f003_t28.txt')
t30 = np.loadtxt('f003_t30.txt')
t32 = np.loadtxt('f003_t32.txt')
t34 = np.loadtxt('f003_t34.txt')
t36 = np.loadtxt('f003_t36.txt')
t38 = np.loadtxt('f003_t38.txt')
t40 = np.loadtxt('f003_t40.txt')


t = [[t02[:,:]], [t04[:,:]], [t06[:,:]], [t08[:,:]], [t10[:,:]],
     [t12[:,:]], [t14[:,:]], [t16[:,:]], [t18[:,:]], [t20[:,:]],
     [t22[:,:]], [t24[:,:]], [t26[:,:]], [t28[:,:]], [t30[:,:]],
     [t32[:,:]], [t34[:,:]], [t36[:,:]], [t38[:,:]], [t40[:,:]]]


out = grab(t)

f003_dcalcite = out[2] #2.0*out[2] + out[5]*out[4]
f003_dglass = out[6]







#############################
# flush 4% / 10 years       #
# t = 3.14e8 (10 years)     #
#############################

t02 = np.loadtxt('f004_t02.txt')
t04 = np.loadtxt('f004_t04.txt')
t06 = np.loadtxt('f004_t06.txt')
t08 = np.loadtxt('f004_t08.txt')
t10 = np.loadtxt('f004_t10.txt')
t12 = np.loadtxt('f004_t12.txt')
t14 = np.loadtxt('f004_t14.txt')
t16 = np.loadtxt('f004_t16.txt')
t18 = np.loadtxt('f004_t18.txt')
t20 = np.loadtxt('f004_t20.txt')
t22 = np.loadtxt('f004_t22.txt')
t24 = np.loadtxt('f004_t24.txt')
t26 = np.loadtxt('f004_t26.txt')
t28 = np.loadtxt('f004_t28.txt')
t30 = np.loadtxt('f004_t30.txt')
t32 = np.loadtxt('f004_t32.txt')
t34 = np.loadtxt('f004_t34.txt')
t36 = np.loadtxt('f004_t36.txt')
t38 = np.loadtxt('f004_t38.txt')
t40 = np.loadtxt('f004_t40.txt')


t = [[t02[:,:]], [t04[:,:]], [t06[:,:]], [t08[:,:]], [t10[:,:]],
     [t12[:,:]], [t14[:,:]], [t16[:,:]], [t18[:,:]], [t20[:,:]],
     [t22[:,:]], [t24[:,:]], [t26[:,:]], [t28[:,:]], [t30[:,:]],
     [t32[:,:]], [t34[:,:]], [t36[:,:]], [t38[:,:]], [t40[:,:]]]


out = grab(t)

f004_dcalcite = out[2] #2.0*out[2] + out[5]*out[4]
f004_dglass = out[6]










#############################
# flush 5% / 10 years       #
# t = 3.14e8 (10 years)     #
#############################

t02 = np.loadtxt('f005_t02.txt')
t04 = np.loadtxt('f005_t04.txt')
t06 = np.loadtxt('f005_t06.txt')
t08 = np.loadtxt('f005_t08.txt')
t10 = np.loadtxt('f005_t10.txt')
t12 = np.loadtxt('f005_t12.txt')
t14 = np.loadtxt('f005_t14.txt')
t16 = np.loadtxt('f005_t16.txt')
t18 = np.loadtxt('f005_t18.txt')
t20 = np.loadtxt('f005_t20.txt')
t22 = np.loadtxt('f005_t22.txt')
t24 = np.loadtxt('f005_t24.txt')
t26 = np.loadtxt('f005_t26.txt')
t28 = np.loadtxt('f005_t28.txt')
t30 = np.loadtxt('f005_t30.txt')
t32 = np.loadtxt('f005_t32.txt')
t34 = np.loadtxt('f005_t34.txt')
t36 = np.loadtxt('f005_t36.txt')
t38 = np.loadtxt('f005_t38.txt')
t40 = np.loadtxt('f005_t40.txt')


t = [[t02[:,:]], [t04[:,:]], [t06[:,:]], [t08[:,:]], [t10[:,:]],
     [t12[:,:]], [t14[:,:]], [t16[:,:]], [t18[:,:]], [t20[:,:]],
     [t22[:,:]], [t24[:,:]], [t26[:,:]], [t28[:,:]], [t30[:,:]],
     [t32[:,:]], [t34[:,:]], [t36[:,:]], [t38[:,:]], [t40[:,:]]]


out = grab(t)

f005_dcalcite = out[2] #2.0*out[2] + out[5]*out[4]
f005_dglass = out[6]






#############################
# flush 3% / 10 years      #
# t = 3.14e8 (10 years)     #
#############################

t02 = np.loadtxt('f006_t02.txt')
t04 = np.loadtxt('f006_t04.txt')
t06 = np.loadtxt('f006_t06.txt')
t08 = np.loadtxt('f006_t08.txt')
t10 = np.loadtxt('f006_t10.txt')
t12 = np.loadtxt('f006_t12.txt')
t14 = np.loadtxt('f006_t14.txt')
t16 = np.loadtxt('f006_t16.txt')
t18 = np.loadtxt('f006_t18.txt')
t20 = np.loadtxt('f006_t20.txt')
t22 = np.loadtxt('f006_t22.txt')
t24 = np.loadtxt('f006_t24.txt')
t26 = np.loadtxt('f006_t26.txt')
t28 = np.loadtxt('f006_t28.txt')
t30 = np.loadtxt('f006_t30.txt')
t32 = np.loadtxt('f006_t32.txt')
t34 = np.loadtxt('f006_t34.txt')
t36 = np.loadtxt('f006_t36.txt')
t38 = np.loadtxt('f006_t38.txt')
t40 = np.loadtxt('f006_t40.txt')


t = [[t02[:,:]], [t04[:,:]], [t06[:,:]], [t08[:,:]], [t10[:,:]],
     [t12[:,:]], [t14[:,:]], [t16[:,:]], [t18[:,:]], [t20[:,:]],
     [t22[:,:]], [t24[:,:]], [t26[:,:]], [t28[:,:]], [t30[:,:]],
     [t32[:,:]], [t34[:,:]], [t36[:,:]], [t38[:,:]], [t40[:,:]]]


out = grab(t)

f006_dcalcite = out[2] #2.0*out[2] + out[5]*out[4]
f006_dglass = out[6]






f006_dcalcite[1] = f006_dcalcite[0]
f005_dcalcite[1] = f005_dcalcite[0]
f004_dcalcite[1] = f004_dcalcite[0]
f003_dcalcite[1] = f003_dcalcite[0]
f002_dcalcite[1] = f002_dcalcite[0]
f001_dcalcite[1] = f001_dcalcite[0]



#print dcalcite_230


fig=plt.figure()

##ax = plt.subplot(2,2,1)
##
##plt.plot(temps, f005_dcalcite, label='5%')
##plt.plot(temps, f004_dcalcite, label='4%')
##plt.plot(temps, f003_dcalcite, label='3%')
##plt.plot(temps, f002_dcalcite, label='2%')
##plt.plot(temps, f001_dcalcite, label='1%')
##
##
##
##plt.title('calcite formation rate')
##handles, labeling = ax.get_legend_handles_labels()
##plt.legend(handles[::-1], labeling[::-1])
##plt.legend(handles, labeling,loc='best',prop={'size':8}, ncol=2)
##
##
##
##
##
##
##
##ax = plt.subplot(2,2,2)
##
##plt.plot(temps, f004_dglass, label='4%')
##plt.plot(temps, f003_dglass, label='3%')
##plt.plot(temps, f002_dglass, label='2%')
##plt.plot(temps, f001_dglass, label='1%')
##
##
##
##plt.title('basalt alteration rate')
##handles, labeling = ax.get_legend_handles_labels()
##plt.legend(handles[::-1], labeling[::-1])
##plt.legend(handles, labeling,loc='best',prop={'size':8}, ncol=2)


ax = plt.subplot(1,1,1)


flush = np.array([.06, .05, .04, .03, .02, .01])


grid = np.array([f006_dcalcite, f005_dcalcite,
                 f004_dcalcite,
                 f003_dcalcite,
                 f002_dcalcite, f001_dcalcite])
grid = grid/np.max(grid)


p = plt.contourf(flush, temps, np.transpose(grid), 18,
               cmap=cm.Spectral_r)

plt.xticks([0.01, 0.02, 0.03, 0.04, 0.05, .06])
ax.set_xticklabels(['.01%','.02%','.03%','.04%','.05%','.06%'])
#plt.yticks(np.arange(0,40,4))

plt.xlabel('RATE OF SEAWATER INCORPORATION [% fluid volume / year]')
plt.ylabel('FLUID TEMPERATURE [$^{\circ}$C]')

# more temperature dependence at higher seawater mixing rates

plt.gca().invert_yaxis()



cbar= plt.colorbar(p, orientation='horizontal')
cbar.ax.set_xlabel('NORMALIZED CaCO$_3$ FORMATION RATE [mol / yr]')


plt.savefig('prelimGrid0.png')

print f004_dcalcite - f005_dcalcite
