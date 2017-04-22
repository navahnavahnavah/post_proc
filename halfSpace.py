import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import streamplot as sp

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams['xtick.major.size'] = 0
plt.rcParams['ytick.major.size'] = 0

# half space
T_o = 2.0 # ocean bottom temp
T_m = 1350.0 # upper mantle temp
z = 400.0 # m down
kappa = 8.0e-7 # m^2 s^-1, thermal diffusivity of lithosphere

def temp(t):
    temp = (T_m - T_o) * math.erf( z / (2.0 * math.sqrt(kappa * t) )) + T_o
    #print  z / (2.0 * math.sqrt(kappa * t) )
    return temp

def round_to_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

halfSpace = np.zeros(600)
for i in range(1,601):
    #print temp(.5*3.14e12*i) ,
    halfSpace[i-1] = temp(.5*3.14e12*i)


sat = np.loadtxt('alterbox.txt')
sat = sat[:300,3:-1]

j = 10

fig=plt.figure()


#xx = times+1.0





ax1=fig.add_subplot(1,1,1,aspect=50.0)



# PLOT W.R.T. temp
ax1.plot(sat[2,:],(sat[28,:]-np.abs(np.min(sat[28,:])))/np.ptp(sat[28,:]),
         label='Anhydrite')
ax1.plot(sat[2,:],(sat[30,:]-np.abs(np.min(sat[30,:])))/np.ptp(sat[30,:]),
         label='Calcite')
ax1.plot(sat[2,:],(sat[17,:]-np.abs(np.min(sat[17,:])))/np.ptp(sat[17,:]),
         label='Kaolinite')
ax1.plot(sat[2,:],(sat[35,:]-np.abs(np.min(sat[35,:])))/np.ptp(sat[35,:]),
         label='Mg-Nontronite')
ax1.plot(sat[2,:],(sat[32,:]-np.abs(np.min(sat[32,:])))/np.ptp(sat[32,:]),
         label='Potassium feldspar')
ax1.plot(sat[2,:],(sat[19,:]-np.abs(np.min(sat[19,:])))/np.ptp(sat[19,:]),
         label='Mg-saponite')


handles, labels = ax1.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])
plt.legend(handles, labels,loc='best',prop={'size':6}, ncol=1)

ax1.set_xlabel('fluid temperature [C]')
plt.ylabel('MINERAL SATURATION INDEX')

tempTicks = np.linspace(np.max(sat[2,:]),np.min(sat[2,:]),10)

ax1.set_xticks(np.round(tempTicks))
#ax1.set_xticklabels(theXtickLabels)
ax1.invert_xaxis()

scaled = np.zeros(21)
for i in range(21):
    #print i*len(sat[0,:])/21
    scaled[i] = sat[0,i*len(sat[0,:])/21]
    print scaled[i]/(3.14e12)

hh = 1.55
ax1.axvline(sat[2,find_nearest(sat[0,:],3.14e13)], color='k') #1myr
ax1.text(54,hh, "1.5 Myr",fontsize=8, rotation=270)
ax1.axvline(sat[2,find_nearest(sat[0,:],6.28e13)], color='k') #2myr
ax1.text(47,hh, "2 Myr",fontsize=8, rotation=270)
ax1.axvline(sat[2,find_nearest(sat[0,:],1.57e13)], color='k') #.5myr
ax1.text(89,hh, ".5 Myr",fontsize=8, rotation=270)
ax1.axvline(sat[2,find_nearest(sat[0,:],4.71e13)], color='k') #1.5myr
ax1.text(64,hh, "1 Myr",fontsize=8, rotation=270)
ax1.axvline(sat[2,find_nearest(sat[0,:],20.0*3.14e13)], color='k') #20myr
ax1.text(17,hh, "20 Myr",fontsize=8, rotation=270)
ax1.axvline(sat[2,find_nearest(sat[0,:],10.0*3.14e13)], color='k') #10myr
ax1.text(27,hh, "10 Myr",fontsize=8, rotation=270)
ax1.axvline(sat[2,find_nearest(sat[0,:],5.0*3.14e13)], color='k') #5myr
ax1.text(31,hh, "5 Myr",fontsize=8, rotation=270)


#ax3=fig.add_subplot(2,1,2)




#plt.xticks(theX2ticks, theX2tickLabels)
plt.title('COOLING OF FLUID ACCORDING TO COOLING LITHOSPHERE')



plt.xlabel('crust age [Myr]')
plt.ylabel('temp')

plt.subplots_adjust(hspace=.5, wspace=.5)

plt.savefig('saturations20.png')
