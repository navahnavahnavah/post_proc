#plotGlass.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from scipy.optimize import curve_fit


outPlag = np.loadtxt('outPlagioclase.txt')
outAug = np.loadtxt('outAugite.txt')
outPig = np.loadtxt('outPigeonite.txt')

out = np.loadtxt('testMat.txt')

def func(x, a, b, c, d, e):
    return a + b*x + c/x + d*np.log10(x) + e/(x*x)
x = outPlag[:,0]+273.0
yn = outPlag[:,1]
popt, pcov = curve_fit(func, x, yn)
print popt
yn = outAug[:,1]
popt, pcov = curve_fit(func, x, yn)
print popt
yn = outPig[:,1]
popt, pcov = curve_fit(func, x, yn)
print popt

print outPlag[11,1]
print outAug[11,1]
print outPig[11,1]

fig=plt.figure()

plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8)

# 3 stillbite 832.2
mStil = 832.2
# 5 sio2(am) 60.0
mSi = 60.0
# 7 kaolinite 258.2
mKao = 258.2
# 9 albite
mAlb = 262.3
# 11 saponite-mg
mSap = 385.537
# 13 celadonite
mCel = 396.8
# 15 Clinoptilolite-Ca
mClin = 1344.4919
# 17 pyrite
mPyr = 120.0
# 19 hematite
mHem = 103.8
# 21 goethite
mGoe = 88.8
# 23 dolomite
mDol = 184.3
# 25 Smectite-high-Fe-Mg
mSmec = 425.685
# 27 dawsonite
mDaws = 144.0
# 29 magnesite
mMagn = 84.3
# 31 siderite
mSid = 115.8
# 33 calcite
mCal = 100.0

mass = out[:,35]*270.0 + out[:,37]*230.0 + out[:,39]*238.6 +  \
       out[:,41]*231.0 + out[:,43]*46.5 + \
       out[:,3]*mStil + out[:,5]*mSi + out[:,7]*mKao + out[:,9]*mAlb +\
       out[:,11]*mSap + out[:,13]*mCel +\
       out[:,15]*mClin + out[:,17]*mPyr + out[:,19]*mHem + out[:,21]*mGoe +\
       out[:,23]*mDol + out[:,25]*mSmec + out[:,27]*mDaws + \
       out[:,29]*mMagn + out[:,31]*mSid + out[:,33]*mCal

#mass = 1.0
out[:,0] = out[:,0]/(3.14e7)

print "phi"
print out[:,45]

print "s_sp"
print out[:,46]

print "water_volume"
print out[:,47]

print "rho_s"
print out[:,48]

################
# precipitates #
################

ax = plt.subplot(2,2,3)
this = out[:,1]
p3, = ax.plot(out[:,0],out[:,3]*mStil/mass, 'b:', label='Stilbite',linewidth=1)
p3, = ax.plot(out[:,0],out[:,5]*mSi/mass, 'b', label='SiO2',linewidth=1)
p3, = ax.plot(out[:,0],out[:,7]*mKao/mass, 'k', label='Kaolinite',linewidth=1)
p3, = ax.plot(out[:,0],out[:,9]*mAlb/mass, 'g', label='Albite',linewidth=1)
p3, = ax.plot(out[:,0],out[:,11]*mSap/mass, 'm--', label='Saponite-Mg',linewidth=1)
p3, = ax.plot(out[:,0],out[:,13]*mCel/mass, 'r', label='Celadonite',linewidth=1)
p3, = ax.plot(out[:,0],out[:,15]*mClin/mass, 'm', label='Clinoptilolite-Ca',linewidth=1)
p3, = ax.plot(out[:,0],out[:,17]*mPyr/mass, 'r--', label='Pyrite',linewidth=1)
p3, = ax.plot(out[:,0],out[:,19]*mHem/mass, 'gold', label='Hematite',linewidth=1)
##p3, = ax.plot(out[:,0],out[:,21]*230.0/mass, 'k:', label='',linewidth=1)
##p3, = ax.plot(out[:,0],out[:,23]*230.0/mass, 'k:', label='',linewidth=1)
##p3, = ax.plot(out[:,0],out[:,25]*230.0/mass, 'k:', label='',linewidth=1)
##p3, = ax.plot(out[:,0],out[:,27]*230.0/mass, 'k:', label='',linewidth=1)
##p3, = ax.plot(out[:,0],out[:,29]*230.0/mass, 'k:', label='',linewidth=1)
#p3, = ax.plot(out[:,0],out[:,33]*230.0/mass, 'k', label='calcite',linewidth=3)


handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

plt.ylabel('MASS FRACTION',fontsize=8)
plt.xlabel('TIME',fontsize=8)
plt.legend(handles, labels,loc=2,prop={'size':6}, ncol=2)



###############
# dissolution #
###############

ax = plt.subplot(2,2,2)




p3, = ax.plot(out[:,0],out[:,35]*270.0/mass, 'r', label='Plagioclase',linewidth=2)
p3, = ax.plot(out[:,0],out[:,37]*230.0/mass, 'g', label='Augite',linewidth=2)
p3, = ax.plot(out[:,0],out[:,39]*238.6/mass, 'm--', label='Pigeonite',linewidth=2)
p3, = ax.plot(out[:,0],out[:,41]*231.0/mass, 'gold', label='Magnetite',linewidth=2)
p3, = ax.plot(out[:,0],out[:,43]*46.5/mass, 'k', label='Basaltic Glass',linewidth=2)
plt.grid(True)



handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])
plt.yticks(np.arange(0,.5,.05))
plt.ylabel('MASS FRACTION',fontsize=8)
plt.xlabel('TIME',fontsize=8)
plt.legend(handles, labels,loc='best',prop={'size':6}, ncol=2)

###########
# pH plot #
###########

ax = plt.subplot(2,2,1)
this = out[:,1]
p3, = ax.plot(out[:,0],out[:,1], 'r', label='pH',linewidth=2)
plt.grid(True)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

plt.ylim(3,8)
plt.yticks(np.arange(3,8,.5))
plt.ylabel('pH',fontsize=8)
plt.xlabel('TIME',fontsize=8)
plt.legend(handles, labels,loc='best',prop={'size':8})




################
#  carbonates  #
################

ax = plt.subplot(2,2,4)
this = out[:,1]
p3, = ax.plot(out[:,0],out[:,31]*mSid/mass, 'm--', label='Siderite',linewidth=1)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

plt.ylabel('MASS FRACTION',fontsize=8)
plt.xlabel('TIME',fontsize=8)
plt.legend(handles, labels,loc='best',prop={'size':8})


plt.savefig('glass1114.png')
