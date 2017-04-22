#The point is to plot my limit on outgassing for a planet to be able
#to maintain habitable conditions at the outer edge of the habitable
#zone. (dsa, 3/18/16)

import numpy as np
from scipy import integrate
from matplotlib.pylab import *

class Dummy:
    pass

def get_outer( params ):
    lV = (params.Ti-params.T0)*(params.k+params.a*params.beta/params.b)+(params.S0-params.Ss)*(1-params.alph_o)*params.beta/params.b/4.0
    return np.exp(lV)

def get_outer_line( params ):
    k = -params.lvw/(params.T0-params.Ti)+(params.beta/params.b)*((params.S0-params.Ss)*(1-params.alph_o)/(params.T0-params.Ti)/4.0-params.a)
    return k

params = Dummy()

params.S0     = 1365.0  #W m-2
params.Ss     = params.S0/1.7**2  #W m-2
params.T0     = 15.0    #C
params.k      = 0.10    #C-1
params.a      = 2.0     #W m-2 C-1
params.b      = 10.0    #W m-2
params.beta   = 0.5     #
params.W0     = 7.0     #bars Gyr-1
params.P0     = 3e-4    #bars
params.alph_o = 0.3     #
params.alph_i = 0.7     #
params.Ti     = -10.0   #C

params.lvw = np.log(10) #log(V/W0)


beta_arr = np.linspace(0.,1.,11)
N        = np.size(beta_arr)
k_arr1   = np.zeros(N)
k_arr2   = np.zeros(N)
k_arr3   = np.zeros(N)

lvw1     = np.log(0.1)
lvw2     = 0.
lvw3     = np.log(10.)

k = 0
while k<N:
    params.beta = beta_arr[k]
    params.lvw = lvw1
    k_arr1[k] = get_outer_line( params )
    params.lvw = lvw2
    k_arr2[k] = get_outer_line( params )
    params.lvw = lvw3
    k_arr3[k] = get_outer_line( params )
    k += 1

fig = figure()
plot(beta_arr,k_arr1,'k',beta_arr,k_arr2,'k',beta_arr,k_arr3,'k',0.5,0.1,'ro')
xlabel(r'$\beta$')
ylabel(r'k [C$^{-1}$]')
title('Threshold for a safe habitable zone')
ylim([0,0.3])
xlim([0,0.8])
text(0.07,0.11,r'V=0.1*W$_0$')
text(0.07,0.02,r'V=W$_0$')
text(0.29,0.02,r'V=10*W$_0$')
text(0.51,0.09,'Default Values',color='r')
text(0.12,0.25,'Habitable Zone Safe')
arrow(0.2, 0.2, -0.1, 0.06, head_width=0.01, head_length=0.01, fc='k', ec='k')

annotate('Habitable Zone Safe', xy=(0.3, 0.24), xytext=(0.1, 0.28), arrowprops=dict(arrowstyle='<|-',fc='k'),)


bbox_props = dict(boxstyle="larrow,pad=0.4", fc="w", ec="k", lw=2.5)
text(.13, .24, "Habitable Zone Safe", weight='bold', ha="center", va="center", rotation='-47',bbox=bbox_props)
fig.savefig('outer_outgassing.jpg')
