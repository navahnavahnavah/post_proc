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
plt.rcParams['xtick.major.pad'] = 3
plt.rcParams['ytick.major.pad'] = 3
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.color_cycle'] = "#CE1836, #F85931, #EDB92E, #A3A948, #009989"

print "doing something..."

sec = ['zero', 'stilbite', 'sio2', 'kaolinite', 'albite', 'saponite', 'celadonite',
       'clinoptilolite', 'pyrite', 'mont_na', 'goethite', 'dolomite', 'smectite',
       'dawsonite', 'magnesite', 'siderite', 'calcite', 'quartz', 'kspar', 'saponite_na',
       'nont_na', 'nont_mg', 'nont_k', 'nont_h', 'nont_ca', 'muscovite', 'mesolite',
       'hematite', 'diaspore', 'akermanite', 'analcime', 'annite', 'clinozoisite',
        'dicalcium_silicate', 'diopside', 'epidote', 'ettringite', 'ferrite_ca', 'foshagite',
        'gismondine', 'gyrolite', 'hedenbergite', 'hillebrandite', 'larnite', 'laumontite',
        'lawsonite', 'merwinite', 'monticellite', 'natrolite', 'okenite', 'phlogopite',
        'prehnite', 'pseudowollastonite', 'rankinite', 'scolecite', 'tobermorite_9a',
        'tremolite', 'wollastonite', 'xonotlite', 'zoisite', 'andradite', 'troilite',
        'pyrrhotite', 'minnesotaite', 'fayalite', 'daphnite_7a', 'daphnite_14a',
        'cronstedtite_7a', 'greenalite', 'aragonite']

##############
# INITIALIZE #
##############

cell = 2
#steps = 400
steps = 50

path = "output/all_diff_only/15_x/"
#path = ""


# load output
x0 = np.loadtxt(path + 'x.txt',delimiter='\n')
y0 = np.loadtxt(path + 'y.txt',delimiter='\n')

# format plotting geometry
x=x0
y=y0
bits = len(x)

xCell = x0
yCell = y0
xCell = xCell[::cell]
yCell= yCell[::cell]
xCell = np.append(xCell, np.max(xCell)+xCell[1])
yCell = np.append(yCell, np.max(yCell)-yCell[-1])
bitsC = len(xCell)

print xCell
print yCell

xg, yg = np.meshgrid(x[:],y[:])


# load output
psi0 = np.loadtxt(path + 'psiMat.txt')

perm0 = np.loadtxt(path + 'permMat.txt')
perm0 = np.log10(perm0)
temp0 = np.loadtxt(path + 'hMat.txt')
glass0 = np.loadtxt(path + 'pri_glass.txt')
geo1 = np.loadtxt(path + 'sec8.txt')
geo16 = np.loadtxt(path + 'sec16.txt')
geo5 = np.loadtxt(path + 'sec19.txt') + np.loadtxt(path + 'sec5.txt')
geo6 = np.loadtxt(path + 'sec6.txt')
geo48 = np.loadtxt(path+'sec20.txt')+np.loadtxt(path+'sec24.txt')+ \
    np.loadtxt(path+'sec26.txt')
geo10 = np.loadtxt(path + 'sec10.txt')

where_are_NaNs = np.isinf(geo1)
geo1[where_are_NaNs] = 0.7

#limit = np.where(geo0 >= .0006)
#geo0[limit] = 0.0

def plotMin(step,matrix,nlabel,colormap,outline):
    p5_range = np.linspace(np.min(matrix[np.nonzero(matrix)]),
                           np.max(matrix[np.nonzero(matrix)]/1.0),levs)
    p5 = plt.contourf(xCell,yCell,step, p5_range,cmap=colormap,alpha=alph,antialiased=True)
    p5l = plt.contour(xCell,yCell,step, p5_range,colors=outline,linewidths=2.0)
    p5l.collections[0].set_label(nlabel)
    return


 
for i in range(0,steps,10): 

    #######################
    # FORMAT GRIDDED DATA #
    #######################
    
    ##    h = h0[i*len(y)-i:((i)*len(y)+len(x))-i-1,:]
    ##    h = np.append(h, h[-1:,:], axis=0)
    ##    h = np.append(h, h[:,-1:], axis=1)

    psi = psi0[i*len(y):((i)*len(y)+len(x)),:]

    perm = perm0[i*len(y):((i)*len(y)+len(x)),:]

    temp = temp0[i*len(y):((i)*len(y)+len(x)),:]

    glass = glass0[(i*len(y0)/cell):(i*len(y0)/cell+len(y0)/cell)-1,:]
    glass = np.append(glass, glass[-1:,:], axis=0)
    glass = np.append(glass, glass[:,-1:], axis=1)

    geo_1 = geo1[(i*len(y0)/cell):(i*len(y0)/cell)+len(y0)/cell,:]
    geo_1 = np.append(geo_1, geo_1[-1:,:], axis=0)
    geo_1 = np.append(geo_1, geo_1[:,-1:], axis=1)

    geo_5 = geo5[(i*len(y0)/cell):(i*len(y0)/cell)+len(y0)/cell,:]
    geo_5 = np.append(geo_5, geo_5[-1:,:], axis=0)
    geo_5 = np.append(geo_5, geo_5[:,-1:], axis=1)


    geo_16 = geo16[(i*len(y0)/cell):(i*len(y0)/cell)+len(y0)/cell,:]
    geo_16 = np.append(geo_16, geo_16[-1:,:], axis=0)
    geo_16 = np.append(geo_16, geo_16[:,-1:], axis=1)

    geo_6 = geo6[(i*len(y0)/cell):(i*len(y0)/cell)+len(y0)/cell,:]
    geo_6 = np.append(geo_6, geo_6[-1:,:], axis=0)
    geo_6 = np.append(geo_6, geo_6[:,-1:], axis=1)
    
    geo_48 = geo48[(i*len(y0)/cell):(i*len(y0)/cell)+len(y0)/cell,:]
    geo_48 = np.append(geo_48, geo_48[-1:,:], axis=0)
    geo_48 = np.append(geo_48, geo_48[:,-1:], axis=1)

    geo_10 = geo10[(i*len(y0)/cell):(i*len(y0)/cell)+len(y0)/cell,:]
    geo_10 = np.append(geo_10, geo_10[-1:,:], axis=0)
    geo_10 = np.append(geo_10, geo_10[:,-1:], axis=1)


    #############
    # FULL PLOT #
    #############

    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1, aspect='equal')

    print xCell.shape
    print yCell.shape
    print geo_1.shape
#    pGlass = plt.pcolor(xCell, yCell, geo_1, vmin=np.min(geo1[np.nonzero(geo1)]),
#                         vmax=np.max(geo1[np.nonzero(geo1)]/1.0),
#                         cmap=cm.Spectral_r, linewidth=0.0, color='#444444')
#    cbar= plt.colorbar(pGlass, orientation='horizontal')
##    cbar.ax.set_xlabel('PRECIPITATED CALCITE [mol]')

    levs = 2
    alph = 0.6

    geo_6f = geo_6
    #geo_6f[geo_6f<=np.max(geo6)/10.0] = 0.0
    plotMin(geo_6f,geo6,'celadonite (stage 1)',cm.Greens,'green')

    geo_10f = geo_10
    geo_10f[geo_10f<=np.max(geo_10)/10.0] = 0.0
    plotMin(geo_10f,geo10,'iron oxyhydroxide (stages 1 + 2)',cm.Reds,'red')


    geo_5f = geo_5
    geo_5f[geo_5f<=np.max(geo5)/6.0] = 0.0
    plotMin(geo_5f,geo5,'saponites (stages 2 + 3)',cm.Oranges,'Chocolate')

    geo_1f = geo_1
    geo_1f[geo_1f<=np.max(geo1)/6.0] = 0.0
    plotMin(geo_1f,geo1,'pyrite (stage 3)',cm.Blues,'blue')

    geo_16f = geo_16
    #geo_16f[geo_16f<=np.max(geo16)/10.0] = 0.0
    plotMin(geo_16f,geo16,'calcite (stages 3 + 4)',cm.Purples,'purple')

    geo_48f = geo_48
    geo_48f[geo_48f<=np.max(geo48)/2.0] = 0.0
    plotMin(geo_48f,geo48,'total zeolites (stage 4)',cm.Greys,'black')



    scolor = 'MediumSpringGreen'
    

    theLegend = plt.legend(prop={'size':10},loc=9, bbox_to_anchor=(1.22,1.03),ncol=1)


    #pGlass = plt.contourf(xCell, yCell, geo, np.arange(0.0004,0.00321,0.0004),
    #                    cmap=cm.winter_r, linewidth=1.0, color='black')
    print xg.shape
    print yg.shape
    print psi.shape
    #contoursPsi = np.arange(np.min(psi),
    #                        np.max(psi)+(np.max(psi)-np.min(psi))/10.0,
    #                     (np.max(steady_psi)-np.min(steady_psi))/10.0)

    p = plt.contour(xg,yg,perm,[-13.05,-18],colors='black',linewidths=np.array([4.0]))
    #psi[psi<1e-11] = 0.0
    CS = plt.contour(xg, yg, psi, 10, colors='black',linewidths=np.array([1.0]))
    

    # formatting
    #theTicks = geoContours

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    
    plt.title('' + str((i*.12)+.12) + 'Myr')

    

    plt.plot([x[5*bits/8], x[5*bits/8]], [-1200.0, 0.0],
             linewidth=3.0, color=scolor)
    plt.text(x[5*bits/8]-80.0, 70.0, '504B',
        bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10},fontsize=8)

    plt.plot([x[(5*bits/8)+bits/3], x[(5*bits/8)+bits/3]], [-1200.0, 0.0],
             linewidth=3.0, color=scolor)
    plt.text(x[(5*bits/8 + bits/3)]-80.0, 70.0, '896A',
        bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10},fontsize=8)

    

    plt.savefig(path+'j100_final_'+str(i)+'.png',
                bbox_inches='tight',bbox_extra_artists=(theLegend,))


print "pyrite"
print "504B:" , np.sum(geo_1[bitsC/2:,bitsC/2.0])
print "896A:" , np.sum(geo_1[bitsC/2:,bitsC-3])

print "saponite-mg"
print "504B:" , np.sum(geo_5[bitsC/2:,bitsC/2.0])
print "896A:" , np.sum(geo_5[bitsC/2:,bitsC-3])

print "celadonite"
print "504B:" , np.sum(geo_6[bitsC/2:,bitsC/2.0])
print "896A:" , np.sum(geo_6[bitsC/2:,bitsC-3])

print "calcite"
print "504B:" , np.sum(geo_16[bitsC/2:,bitsC/2.0])
print "896A:" , np.sum(geo_16[bitsC/2:,bitsC-3])




print "ALL DONE!"


xb = 3*bits/4
xa = -4 #5*bits/8

print "504B:" , 1.6*(temp[-1,xb] - temp[-2,xb])/(y[1]-y[0])
print "896A:" , 1.6*(temp[-1,xa] - temp[-2,xa])/(y[1]-y[0])


xx = 5*bits/8



fig=plt.figure()
ax1=fig.add_subplot(1,1,1, aspect=3.0)
plt.plot(x[xx:],-1000.0*2.6*(temp[-1,xx:] - temp[-2,xx:])/(y[1]-y[0]),linewidth=2.0,color='k')


plt.plot([x[5*bits/8], x[5*bits/8]], [180.0, 360.0], linewidth=3.0, color=scolor)
plt.text(x[5*bits/8]-30.0, 365.0, '504B',
        bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10}, fontsize=8)

plt.plot([x[(5*bits/8)+bits/3], x[(5*bits/8)+bits/3]], [180.0, 360.0], linewidth=3.0, color=scolor)
plt.text(x[(5*bits/8 + bits/3)]-30.0, 364.0, '896A',
        bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10}, fontsize=8)

plt.xlabel('LATERAL DISTANCE [m]')
plt.ylabel('HEAT FLOW [mW m$^{-2}$]')
plt.title('HEAT FLOW OUT OF RIFT FLANK')

plt.savefig(path+'j_hf.png',bbox_inches='tight')




# CROSS SECTION
fig=plt.figure()

xca = 5*bitsC/8 + bitsC/3

xcb = 5*bitsC/8

ax1=fig.add_subplot(2,1,1)

# zeolies total
geo_48[geo_48>0.0] = 1.0
plt.scatter(geo_48[:,xcb],yCell,label='total zeolites')
plt.text(3, -100.0, 'zeolites', rotation='vertical')

# celadonite
geo_6[geo_6>0.0] = 2.0
plt.scatter(geo_6[:,xcb],yCell,label='celadonite')

# iron oxyhydroxide
geo_10[geo_10>0.0] = 0.0
plt.scatter(geo_10[:,xcb],yCell,label='iron oxyhydroxide')
plt.text(3, -100.0, 'Fe-hydroxide', rotation='vertical')

# saponite
geo_5[geo_5>0.0] = 4.0
plt.scatter(geo_5[:,xcb],yCell,label='saponites')

# pyrite?
geo_1[geo_1>0.0] = 5.0
plt.scatter(geo_1[:,xcb],yCell,label='pyrite')

# calcite
geo_16[geo_16>0.0] = 6.0
plt.scatter(geo_16[:,xcb],yCell,label='calcite')

plt.ylim([-850.0,-200.0])
plt.title('HOLE 504B')


plt.savefig(path+'j_x_.png',bbox_inches='tight')
