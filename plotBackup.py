import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
plt.rcParams['contour.negative_linestyle'] = 'solid'

#####################
# LOAD MODEL OUTPUT #
#####################

t = np.loadtxt('t1.txt',delimiter='\n')
x0 = np.loadtxt('x1.txt',delimiter='\n')
y0 = np.loadtxt('y1.txt',delimiter='\n')

x=x0
y=y0
bits = len(x)
x = np.append(x0, np.max(x0)+.001)
y = np.append(y0, np.max(y0)+(y0[-1]-y0[-2]))

xg, yg = np.meshgrid(x[:],y[:])

h = np.loadtxt('h1.txt')
u= np.loadtxt('uMat1.txt')
v= np.loadtxt('vMat1.txt')
psi = np.loadtxt('psiMat1.txt')
rho = np.loadtxt('rho1.txt')
viscosity = 1e-3
permeability = np.loadtxt('permeability1.txt')
permeability = permeability

i=3

fig=plt.figure()


i=99

#######################
# MAKE DATA PLOTTABLE #
#######################

h = np.append(h, h[-1:,:], axis=0)
h = np.append(h, h[:,-1:], axis=1)

psi = np.append(psi, psi[-1:,:], axis=0)
psi = np.append(psi, psi[:,-1:], axis=1)

v = np.append(v, v[-1:,:], axis=0)
v = np.append(v, v[:,-1:], axis=1)

u = np.append(u, u[-1:,:], axis=0)
u = np.append(u, u[:,-1:], axis=1)

permeability = np.append(permeability, permeability[-1:,:], axis=0)
permeability = np.append(permeability, permeability[:,-1:], axis=1)

rho = np.append(rho, rho[-1:,:], axis=0)
rho = np.append(rho, rho[:,-1:], axis=1)


####################
# STREAM FUNCTIONS #
####################

ax1=fig.add_subplot(2,1,1, aspect='equal')

p = plt.pcolor(xg,yg,np.log10(permeability),cmap=cm.summer)
print np.gradient(h)[1].shape
print xg.shape
CS = plt.contour(xg, yg, psi, 30, colors='k',linewidths=np.array([1.6]))

plt.title("STREAMFUNCTIONS",fontsize=8)

plt.xlim(np.min(x), np.max(x))
plt.ylim(-1300, np.max(y))

#############
# ISOTHERMS #
#############

ax1=fig.add_subplot(2,1,2, aspect='equal')

CS = plt.contour(xg, yg, h, 35, colors='#660066',linewidths=np.array([2.0]))
plt.clabel(CS, fontsize=9, inline=1,fmt='%3.0f')


plt.title("ISOTHERMS",fontsize=8)

plt.xlim(np.min(x), np.max(x))
plt.ylim(-1300, np.max(y))
#plt.ylim(np.min(y), np.max(y))


    
plt.subplots_adjust(bottom=.2, left=.1, right=.90, top=0.9, hspace=.3)

cax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
cbar = plt.colorbar(p, cax=cax,orientation='horizontal')

uniques = np.unique(np.log10(permeability))
print len(uniques)
cbar = plt.colorbar(p, cax=cax,orientation='horizontal')
cbar.set_label(r'log of permeability',fontsize=8)

plt.savefig('nn14.png')
print "yeah!"
