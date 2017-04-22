# square_wave.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rcParams['axes.titlesize'] = 12

plt.rcParams['axes.color_cycle'] = "#FF0000, #F89931, #EDB92E, #A3A948, #009989"

x = np.linspace(0.0,1.0,101)
y = np.linspace(0.0,1.0,101)
z = np.zeros([len(x),len(y)])
z[5:25,5:25] = 1.0

dx = x[1] - x[0]
dy = y[1] - y[0]
u = 0.035
v = 0.035

t_max = 10.0


def upwind(dt):
    tn = int(t_max/dt)
    print tn
    print "cfl" , u*dt/dx
    upwind = np.zeros([len(x),len(y)])
    upwind = np.zeros([len(x),len(y)])
    upwind[5:25,5:25] = 1.0# + 0.001*np.linspace(0.0,200.0,200)
    for i in range(tn):
        upwind[1:-1,:] = upwind[1:-1,:] - (v*dt/dx)*(upwind[1:-1,:] - upwind[:-2,:])
        upwind[:,1:-1] = upwind[:,1:-1] - (u*dt/dy)*(upwind[:,1:-1] - upwind[:,:-2])
    return upwind
    
upwind_cfl_1 = np.zeros([len(x),len(y)])
upwind_cfl_1 = upwind(10.0*dx/(.35)) 
upwind_cfl_10 = np.zeros([len(x),len(y)])
upwind_cfl_10 = upwind(10.0*dx/(3.5)) 


fig=plt.figure()

ax1=fig.add_subplot(2,2,1)

plt.pcolor(x,y,z)


ax1=fig.add_subplot(2,2,2)

plt.pcolor(x,y,upwind_cfl_1)

ax1=fig.add_subplot(2,2,3)

plt.pcolor(x,y,upwind_cfl_10)



plt.savefig('square_2d.png')