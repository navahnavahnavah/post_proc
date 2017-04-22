# plotPrelim.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import streamplot as sp
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams['xtick.major.size'] = 0
plt.rcParams['ytick.major.size'] = 0
#plt.rcParams['xtick.direction'] = 'out'
#plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.pad'] = 3
plt.rcParams['ytick.major.pad'] = 3
plt.rcParams['axes.linewidth'] = 1.0
#plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"
plt.rcParams['axes.color_cycle'] = "#CC0000, #FF3300, #FF9900, #FFCC00, \
                                    #00FF00, #339900, #009966, #0000FF, \
                                    #6600CC, #990099"


print "doing something..."


#####################
# LOAD MODEL OUTPUT #
#####################


steps = 200
x = np.arange(steps)
output = np.loadtxt('f003_t16.txt')



print output[1,1]

##############
# FIRST PLOT #
##############



labels = ["time", "pH", "pe", "Alk", "C", "C", "Mg", "Na", "K", "Fe", 
"S", "S", "Cl", "Al", "m_HCO3-", "stilbite", "d_stilbite", 
"sio2(am", "d_sio2(am)", "kaolinite", "d_kaolinite", "albite", 
"d_albite", "saponite-m", "d_saponite-mg", "celadonite", 
"d_celadonite", "Clinoptilolite-Ca", "d_Clinoptilolite-Ca", 
"pyrite", "d_pyrite", "Montmor-Na", "d_Montmor-Na", "goethite", 
"d_goethit", "dolomite", "d_dolomite", "Smectite-high-Fe-Mg", 
"d_Smectite-high-Fe-Mg", "Dawsonit", "d_Dawsonite", "magnesite", 
"d_magnesite", "siderite", "d_siderite", "calcite", "d_calcite", 
"quartz", "d_quartz", "k-feldspar", "d_k-feldspar", 
"saponite-n", "d_saponite-na", "Nontronite-Na", 
"d_Nontronite-Na", "Nontronite-Mg", "d_Nontronite-Mg", 
"Nontronite-", "d_Nontronite-K", "Nontronite-H", 
"d_Nontronite-H", "Nontronite-Ca", "d_Nontronite-Ca", 
"muscovit", "d_muscovite", "mesolite", "d_mesolite", 
"hematite", "d_hematite", "diaspor", "d_diaspore", 
"k_plagioclase", "dk_plagioclase", "k_augite", "dk_augite", 
"k_pigeonit", "dk_pigeonite", "k_magnetite", "dk_magnetite", 
"k_bglass", "dk_bglass", "V_R(phi", "V_R(s_sp)", 
"V_R(water_volume)", "V_R(rho_s)"]

print len(labels)

# PRECIPITATES

fig=plt.figure()
ax1=fig.add_subplot(2,2,1)

print x.shape
print output[79,:].shape

for i in range(15,71,2):
    if np.max(output[i,:]) > 0.0:
        p = plt.plot(x, output[i,:], label=labels[i], linewidth=1.0)

handles, labeling = ax1.get_legend_handles_labels()
plt.legend(handles[::-1], labeling[::-1])
plt.legend(handles, labeling,loc='best',prop={'size':8}, ncol=2)


plt.title('PRECIPITATES')


# PRIMARIES

ax1=fig.add_subplot(2,2,2)

print x.shape
print output[79,:].shape


for i in range(71,81,2):
    p = plt.plot(x, output[i,:]/np.max(output[i,:]),
                 label=labels[i], linewidth=1.5)

#p = plt.plot(x, output[4,:], label=labels[4])

handles, labeling = ax1.get_legend_handles_labels()
plt.legend(handles[::-1], labeling[::-1])
plt.legend(handles, labeling,loc='best',prop={'size':8}, ncol=2)


plt.title('PRIMARIES')



# ALKALINITY

ax1=fig.add_subplot(2,2,3)

print x.shape
print output[79,:].shape

#p = plt.plot(x, output[3,:]*output[83,:], label=labels[3], linewidth=1.5)
#p = plt.plot(x, output[4,:]*output[83,:], label=labels[4], linewidth=1.5)

p = plt.plot(x, output[3,:], label=labels[3], linewidth=1.5)
p = plt.plot(x, output[4,:], label=labels[4], linewidth=1.5)

handles, labeling = ax1.get_legend_handles_labels()
plt.legend(handles[::-1], labeling[::-1])
plt.legend(handles, labeling,loc='best',prop={'size':8}, ncol=2)


plt.title('ALKALINITY')



# pH

ax1=fig.add_subplot(2,2,4)

print x.shape
print output[79,:].shape

p = plt.plot(x, output[1,:], label=labels[1], linewidth=1.5)

handles, labeling = ax1.get_legend_handles_labels()
plt.legend(handles[::-1], labeling[::-1])
plt.legend(handles, labeling,loc='best',prop={'size':8}, ncol=2)


plt.title('pH')




plt.savefig('pre13.png')

print output[45,:]
