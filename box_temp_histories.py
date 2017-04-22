# box_temp_histories.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse, Polygon

tn = 1000

outpath = "basalt_box_output/"

col = ['mediumvioletred', 'lime', 'royalblue','orchid', 'g', 'c', 'b', 'purple', 'm', 'hotpink', 'gray']

age = np.linspace(0.0,5.0,tn)
q = np.zeros([tn])
q_ht = np.zeros([tn])
q_2ht = np.zeros([tn])
k_cond = 1.9
temp_sw = 275.0

temp_100 = np.zeros([tn])
temp_200 = np.zeros([tn])
temp_300 = np.zeros([tn])

temp_100_ht = np.zeros([tn])
temp_200_ht = np.zeros([tn])
temp_300_ht = np.zeros([tn])

temp_100_2ht = np.zeros([tn])
temp_200_2ht = np.zeros([tn])
temp_300_2ht = np.zeros([tn])

for i in range(1,len(age)):
    q[i] = 506.7*(age[i]**(-0.5))
    q_ht[i] = 506.7*(age[i]**(-0.5))*(0.5+(.0142*age[i]*0.5))
    q_2ht[i] = 506.7*(age[i]**(-0.5))*(0.25+(.0142*age[i]*0.75))
    temp_100[i] = 2.0 + (q[i]*100.0/(1000.0*1.8))
    temp_200[i] = 2.0 + (q[i]*200.0/(1000.0*1.8))
    temp_300[i] = 2.0 + (q[i]*300.0/(1000.0*1.8))
    
    temp_100_ht[i] = 2.0 + (q_ht[i]*100.0/(1000.0*1.8))
    temp_200_ht[i] = 2.0 + (q_ht[i]*200.0/(1000.0*1.8))
    temp_300_ht[i] = 2.0 + (q_ht[i]*300.0/(1000.0*1.8))
    
    temp_100_2ht[i] = 2.0 + (q_2ht[i]*100.0/(1000.0*1.8))
    temp_200_2ht[i] = 2.0 + (q_2ht[i]*200.0/(1000.0*1.8))
    temp_300_2ht[i] = 2.0 + (q_2ht[i]*300.0/(1000.0*1.8))


fig=plt.figure()

ax1=fig.add_subplot(1,1,1)
ax1.grid(True)


# plt.plot(age,temp_100_ht, label='100m depth',c=col[0],lw=2,linestyle='-')
# plt.plot(age,temp_200_ht, label='100m depth',c=col[1],lw=2,linestyle='-')
# plt.plot(age,temp_300_ht, label='100m depth',c=col[2],lw=2,linestyle='-')



ax1.fill_between(age, temp_100_2ht, temp_100, facecolor='none', alpha=0.8, edgecolor=col[0], interpolate=True, lw=0,zorder=30,hatch='x')

#ax1.fill_between(age, temp_200_2ht, temp_200, facecolor='w', alpha=0.5, edgecolor=None, interpolate=True, lw=0)
ax1.fill_between(age, temp_200_2ht, temp_200, facecolor='none', alpha=0.8, edgecolor=col[1], interpolate=True, lw=0,zorder=20,hatch='x')

#ax1.fill_between(age, temp_300_2ht, temp_200, facecolor='w', alpha=0.5, edgecolor=None, interpolate=True, lw=0)
ax1.fill_between(age, temp_300_2ht, temp_300, facecolor='none', alpha=0.8, edgecolor=col[2], interpolate=True, lw=0,zorder=10,hatch='x')


plt.plot(age,temp_100, label='100m depth',color=col[0],lw=2,linestyle='-',zorder=30)
plt.plot(age,temp_200, label='200m depth',color=col[1],lw=2,linestyle='-',zorder=20)
plt.plot(age,temp_300, label='300m depth',color=col[2],lw=2,linestyle='-',zorder=10)

plt.plot(age,temp_100_2ht,color=col[0],lw=2,linestyle='-',zorder=30)
plt.plot(age,temp_200_2ht,color=col[1],lw=2,linestyle='-',zorder=20)
plt.plot(age,temp_300_2ht,color=col[2],lw=2,linestyle='-',zorder=10)

lgd = plt.legend(fontsize=10,loc='best',ncol=1)
lgd.get_title().set_fontsize(10)
plt.xlabel('Crust Age [Ma]')
plt.ylabel('Reactive Zone Temperature [$^o$C]')
plt.ylim([0.0, 100.0])

plt.savefig(outpath+'box_temp_histories'+'.png')

arr = temp_100_2ht

print "temp_100_ht"

# FORTRAN PRINT
arr[0] = arr[1]
string = ""
for i in range(tn):
    string = string + str(arr[i]) + ", "
    if i % 6 == 0:
        string = string + "&" + "\n" + "& "
print string

#PYTHON PRINT
string = ""
for i in range(tn):
    string = string + str(arr[i]) + ", "
    if i % 6 == 0:
        string = string + "\n"
#print string