# square_wave.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams['axes.titlesize'] = 12
plt.rc('axes',edgecolor='w')

plt.rcParams['axes.color_cycle'] = "#FF0000, #F89931, #EDB92E, #A3A948, #009989"

x = np.linspace(0.0,1.0,1001)
ic_vec = np.zeros(len(x))
print ic_vec[200:300].shape
ic_vec[200:300] = 1.0# + 0.001*np.linspace(0.0,200.0,200)

x0 = np.linspace(0.0,1.0,10001)
ic_vec0 = np.zeros(len(x0))
print ic_vec0[5500:7000].shape
ic_vec0[5500:7000] = 1.0# + 0.001*np.linspace(0.0,200.0,200)


dx = x[1] - x[0]
u = .05
u_vec = 0.05*np.ones(len(x))
# for i in range(len(x)):
#     u_vec[i] = -0.05
#dt = .0002
t_max = 5.0


def upwind(dt):
    tn = int(t_max/dt)
    print tn
    print u*dt/dx
    upwind = np.zeros(len(x))
    upwind_next = np.zeros(len(x))
    upwind_next[250:450] = 1.0# + 0.001*np.linspace(0.0,200.0,200)
    for i in range(tn):
        upwind_next[1:-1] = upwind_next[1:-1] - (u*dt/dx)*(upwind_next[1:-1] - upwind_next[:-2])
    upwind = upwind_next
        # for j in range(1,len(x)-1):
        #     upwind_next[j] = upwind[j] - (u_vec[j]*dt/dx)*(upwind[j+1] - upwind[j])

    return upwind
    

def upwind_cor(dt):
    tn = int(t_max/dt)
    print tn
    print u*dt/dx
    upwind_cor = np.zeros(len(x))
    upwind_cor_next = np.zeros(len(x))
    upwind_cor_next[250:450] = 1.0# + 0.001*np.linspace(0.0,200.0,200)
    sigma1 = 0.0
    sigma2 = 0.0
    sigma1a = np.zeros(len(x))
    sigma1b = np.zeros(len(x))
    sigma2a = np.zeros(len(x))
    sigma2b = np.zeros(len(x))
    upwind_cor = upwind_cor_next
    cor = 0.0
    for i in range(tn):
        
        
        
        sigma1a[2:-1] = (upwind_cor[3:] - upwind_cor[2:-1])/dx      # (i+1) - (i)
        #sigma1a = (upwind_cor_next[3:] - upwind_cor_next[1:-2])/(2.0*dx)
        sigma1b[2:-1] = (upwind_cor[2:-1] - upwind_cor[1:-2])/dx     # (i) - (i-1)

            
        upwind_cor_next[2:-1] = upwind_cor[2:-1] - (u*dt/dx)*(upwind_cor[2:-1] - upwind_cor[1:-2])
        for j in range(1,len(x)):
            if np.abs(sigma1a[j]) < np.abs(sigma1b[j]):
                sigma1 = sigma1a[j]
            if np.abs(sigma1b[j]) < np.abs(sigma1a[j]):
                sigma1 = sigma1b[j]
            if sigma1a[j]*sigma1b[j] <= 0.0:
                sigma1 = 0.0
                
                
            if np.abs(sigma1a[j-1]) < np.abs(sigma1b[j-1]):
                sigma2 = sigma1a[j-1]
            if np.abs(sigma1b[j-1]) < np.abs(sigma1a[j-1]):
                sigma2 = sigma1b[j-1]
            if sigma1a[j-1]*sigma1b[j-1] <= 0.0:
                sigma2 = 0.0

            #if sigma1*sigma2 > 0.0:
            cor = (u*dt/(2.0*dx))*(sigma1 - sigma2)*(dx-(u*dt))
            upwind_cor_next[j] = upwind_cor_next[j] - cor

        
        upwind_cor = upwind_cor_next
    return upwind_cor
    
    
    
def upwind_cor_b(dt):
    tn = int(t_max/dt)
    print tn
    print u*dt/dx
    upwind_cor_b = np.zeros(len(x))
    upwind_cor_b_next = np.zeros(len(x))
    upwind_cor_b_next[250:450] = 1.0# + 0.001*np.linspace(0.0,200.0,200)
    sigma1 = 0.0
    sigma2 = 0.0
    sigma1p = 0.0
    sigma2p = 0.0
    sigma1f = 0.0
    sigma2f = 0.0
    sigma1a = np.zeros(len(x))
    sigma1b = np.zeros(len(x))
    sigma1c = np.zeros(len(x))
    sigma1d = np.zeros(len(x))
    sigma2a = np.zeros(len(x))
    sigma2b = np.zeros(len(x))
    sigma2c = np.zeros(len(x))
    sigma2d = np.zeros(len(x))
    upwind_cor_b= upwind_cor_b_next
    for i in range(tn):
        
        sigma1a[2:-1] = (upwind_cor_b[3:] - upwind_cor_b[2:-1])/dx                 # (i+1) - (i)
        #sigma1a = (upwind_cor_b_next[3:] - upwind_cor_b_next[1:-2])/(2.0*dx)
        sigma1b[2:-1] = 2.0*(upwind_cor_b[2:-1] - upwind_cor_b[1:-2])/dx           # 2.0 * ((i) - (i-1))
        
        sigma1c[2:-1] = 2.0*(upwind_cor_b[3:] - upwind_cor_b[2:-1])/dx             # 2.0 * ((i+1) - (i))
        #sigma1c = 2.0*(upwind_cor_b_next[3:] - upwind_cor_b_next[1:-2])/(2.0*dx)
        sigma1d[2:-1] = (upwind_cor_b[2:-1] - upwind_cor_b[1:-2])/dx               # (i) - (i-1)
        
        
        #
        # sigma2a = 2.0*(upwind_cor_b[2:-1] - upwind_cor_b[1:-2])/dx           # 2.0 * ((i) - (i-1))
        # #sigma2b = (upwind_cor_b_next[2:-1] - upwind_cor_b_next[:-3])/(2.0*dx)
        # sigma2b = (upwind_cor_b[1:-2] - upwind_cor_b[:-3])/dx                # (i-1) - (i-2)
        #
        # sigma2c = (upwind_cor_b[2:-1] - upwind_cor_b[1:-2])/dx               # (i) - (i-1)
        # #sigma2d = 2.0*(upwind_cor_b_next[2:-1] - upwind_cor_b_next[:-3])/(2.0*dx)
        # sigma2d = 2.0*(upwind_cor_b[1:-2] - upwind_cor_b[:-3])/dx            # 2.0*((i-1) - (i-2))
            
        upwind_cor_b_next[2:-1] = upwind_cor_b[2:-1] - (u*dt/dx)*(upwind_cor_b[2:-1] - upwind_cor_b[1:-2])
        for j in range(1,len(x)):
            if np.abs(sigma1a[j]) < np.abs(sigma1b[j]):
                sigma1 = sigma1a[j]
            if np.abs(sigma1b[j]) < np.abs(sigma1a[j]):
                sigma1 = sigma1b[j]
            # if sigma1a[j]*sigma1b[j] <= 0.0:
            #     sigma1 = 0.0
                
            if np.abs(sigma1c[j]) < np.abs(sigma1d[j]):
                sigma1p = sigma1c[j]
            if np.abs(sigma1d[j]) < np.abs(sigma1c[j]):
                sigma1p = sigma1d[j]
            # if sigma1c[j]*sigma1d[j] <= 0.0:
            #     sigma1p = 0.0
                
            if np.abs(sigma1a[j-1]) < np.abs(sigma1b[j-1]):
                sigma2 = sigma1a[j-1]
            if np.abs(sigma1b[j-1]) < np.abs(sigma1a[j-1]):
                sigma2 = sigma1b[j-1]
            # if sigma1a[j-1]*sigma1b[j-1] <= 0.0:
            #     sigma2 = 0.0
                
            if np.abs(sigma1c[j-1]) < np.abs(sigma1d[j-1]):
                sigma2p = sigma1c[j-1]
            if np.abs(sigma1d[j-1]) < np.abs(sigma1c[j-1]):
                sigma2p = sigma1d[j-1]
            # if sigma2a[j-1]*sigma2b[j-1] <= 0.0:
            #     sigma2p = 0.0

                
            if np.abs(sigma1) > np.abs(sigma1p):
                sigma1f = sigma1
            if np.abs(sigma1) < np.abs(sigma1p):
                sigma1f = sigma1p
            if sigma1*sigma1p <= 0.0:
                sigma1f = 0.0
                
            if np.abs(sigma2) > np.abs(sigma2p):
                sigma2f = sigma2
            if np.abs(sigma2) < np.abs(sigma2p):
                sigma2f = sigma2p
            if sigma2*sigma2p <= 0.0:
                sigma2f = 0.0

            
            #if sigma1f*sigma2f > 0.0:
            cor = (u*dt/(2.0*dx))*(sigma1f - sigma2f)*(dx-(u*dt))
            upwind_cor_b_next[j] = upwind_cor_b_next[j] - cor

        
        upwind_cor_b = upwind_cor_b_next
    return upwind_cor_b
    

upwind_cfl_10 = upwind(.002)
#upwind_cfl_25 = upwind(.005)
upwind_cfl_50 = upwind(.01)
upwind_cfl_95 = upwind(.019)
upwind_cfl_1 = upwind(.02)

upwind_cor_cfl_10 = upwind_cor(.002)
upwind_cor_cfl_50 = upwind_cor(.01)
upwind_cor_cfl_95 = upwind_cor(.019)
upwind_cor_cfl_1 = upwind_cor(.02)

#
upwind_cor_b_cfl_10 = upwind_cor_b(.002)
upwind_cor_b_cfl_50 = upwind_cor_b(.01)
upwind_cor_b_cfl_95 = upwind_cor_b(.019)
upwind_cor_b_cfl_1 = upwind_cor_b(.02)


# upwind_cor_b_cfl_10 = np.zeros(len(x))
# upwind_cor_b_cfl_50 = np.zeros(len(x))
#upwind_cor_b_cfl_95 = np.zeros(len(x))
# upwind_cor_b_cfl_1 = np.zeros(len(x))

print "max thing .95"
print np.max(np.abs(upwind_cor_cfl_95 - upwind_cfl_95))
print "max thing .5"
print np.max(np.abs(upwind_cor_cfl_50 - upwind_cfl_50))
print "max thing .1"
print np.max(np.abs(upwind_cor_cfl_10 - upwind_cfl_10))
#
print "max thing b .95"
print np.max(np.abs(upwind_cor_b_cfl_95 - upwind_cfl_95))
print "max thing b .5"
print np.max(np.abs(upwind_cor_b_cfl_50 - upwind_cfl_50))
print "max thing b .1"
print np.max(np.abs(upwind_cor_b_cfl_10 - upwind_cfl_10))
lax = np.zeros(len(x))



fig=plt.figure(facecolor='black')


ax1=fig.add_subplot(2,2,1)

ax1.set_axis_bgcolor('black')

plt.plot(x0,ic_vec0,'w',label='actual solution',lw=1.0)
plt.plot(x0,ic_vec0,'#0099FF',label='upwind + corrections',lw=1.5)
plt.title('CFL = 1.00', color = 'w')

legend = plt.legend(fontsize=8,ncol=2,loc='upper center')
legend.get_frame().set_facecolor('black')
for text in legend.get_texts():
    plt.setp(text, color = 'w')

ax1.spines['bottom'].set_color('white')
# ax1.spines['top'].set_color('white')
ax1.spines['left'].set_color('white')
# ax1.spines['right'].set_color('white')
ax1.xaxis.label.set_color('white')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

plt.ylim(-0.2,1.7)
plt.xlim(0.67,0.73)



ax1=fig.add_subplot(2,2,2)

ax1.set_axis_bgcolor('black')

plt.plot(x0,ic_vec0,'w',label='actual solution')
plt.plot(x,upwind_cfl_95,'limegreen',label='upwind',lw=1.0)
plt.plot(x,upwind_cor_cfl_95,'limegreen',label='minmod correction',lw=1.5,linestyle=':')
plt.plot(x,upwind_cor_b_cfl_95,'limegreen',label='superbee correction',lw=1.5,linestyle='--')
plt.title('CFL = 0.95', color = 'w')

legend = plt.legend(fontsize=8,ncol=2,loc='upper center')
legend.get_frame().set_facecolor('black')
for text in legend.get_texts():
    plt.setp(text, color = 'w')

ax1.spines['bottom'].set_color('white')
# ax1.spines['top'].set_color('white')
ax1.spines['left'].set_color('white')
# ax1.spines['right'].set_color('white')
ax1.xaxis.label.set_color('white')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

plt.ylim(-0.2,1.7)
plt.xlim(0.67,0.73)



ax1=fig.add_subplot(2,2,3)

ax1.set_axis_bgcolor('black')

plt.plot(x0,ic_vec0,'w',label='actual solution')
plt.plot(x,upwind_cfl_50,'orange',label='upwind',lw=1.0)
plt.plot(x,upwind_cor_cfl_50,'orange',label='minmod correction',lw=1.5,linestyle=':')
plt.plot(x,upwind_cor_b_cfl_50,'orange',label='superbee correction',lw=1.5,linestyle='--')
plt.title('CFL = 0.50', color = 'w')

legend = plt.legend(fontsize=8,ncol=2,loc='upper center')
legend.get_frame().set_facecolor('black')
for text in legend.get_texts():
    plt.setp(text, color = 'w')

ax1.spines['bottom'].set_color('white')
# ax1.spines['top'].set_color('white')
ax1.spines['left'].set_color('white')
# ax1.spines['right'].set_color('white')
ax1.xaxis.label.set_color('white')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

plt.ylim(-0.2,1.7)
plt.xlim(0.67,0.73)



ax1=fig.add_subplot(2,2,4)

ax1.set_axis_bgcolor('black')

plt.plot(x0,ic_vec0,'w',label='actual solution')
plt.plot(x,upwind_cfl_10,'r',label='upwind',lw=1.0)
plt.plot(x,upwind_cor_cfl_10,'r',label='minmod correction',lw=1.5,linestyle=':')
plt.plot(x,upwind_cor_b_cfl_10,'r',label='superbee correction',lw=1.5,linestyle='--')
plt.title('CFL = 0.10', color = 'w')

legend = plt.legend(fontsize=8,ncol=2,loc='upper center')
legend.get_frame().set_facecolor('black')
for text in legend.get_texts():
    plt.setp(text, color = 'w')

ax1.spines['bottom'].set_color('white')
# ax1.spines['top'].set_color('white')
ax1.spines['left'].set_color('white')
# ax1.spines['right'].set_color('white')
ax1.xaxis.label.set_color('white')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

plt.ylim(-0.2,1.7)
plt.xlim(0.67,0.73)


plt.savefig('square_wave.png',facecolor=fig.get_facecolor())