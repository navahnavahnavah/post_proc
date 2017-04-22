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
steps =8

path = "output/costaRica/"
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


xb = 34*bits/50 - bits/5
xa = 34*bits/50

xcb = 34*bitsC/50 - bitsC/5
xca = 34*bitsC/50




# load output
psi0 = np.loadtxt(path + 'psiMat.txt')

perm0 = np.loadtxt(path + 'permMat.txt')
perm0 = np.log10(perm0)
temp0 = np.loadtxt(path + 'hMat.txt')
#temp0[temp0<273.0] = 273.0
temp0 = temp0 - 273.0

ca_e0 = np.loadtxt(path + 'sol_ca.txt')
mg_e0 = np.loadtxt(path + 'sol_mg.txt')



#limit = np.where(geo0 >= .0006)
#geo0[limit] = 0.0

def plotMin(step,matrix,nlabel,colormap,outline):
    p5_range = np.linspace(np.min(matrix[np.nonzero(matrix)]),
                           np.max(matrix[np.nonzero(matrix)]/1.0),levs)
    p5 = plt.contourf(xCell,yCell,step, p5_range,cmap=colormap,alpha=alph,antialiased=True)
    p5l = plt.contour(xCell,yCell,step, p5_range,colors=outline,linewidths=2.0)
    p5l.collections[0].set_label(nlabel)
    return

def cut(geo0,index):
    geo_cut = geo0[(index*len(y0)/cell):(index*len(y0)/cell+len(y0)/cell),:]
    geo_cut = np.append(geo_cut, geo_cut[-1:,:], axis=0)
    geo_cut = np.append(geo_cut, geo_cut[:,-1:], axis=1)
    return geo_cut

 
for i in range(0,steps,1): 

    #######################
    # FORMAT GRIDDED DATA #
    #######################
    
    ##    h = h0[i*len(y)-i:((i)*len(y)+len(x))-i-1,:]
    ##    h = np.append(h, h[-1:,:], axis=0)
    ##    h = np.append(h, h[:,-1:], axis=1)

    psi = psi0[i*len(y):((i)*len(y)+len(x)),:]
    perm = perm0[i*len(y):((i)*len(y)+len(x)),:]
    temp = temp0[i*len(y):((i)*len(y)+len(x)),:]


    ca_e = cut(ca_e0,i)
    mg_e = cut(mg_e0,i)
    



    #############
    # HEAT PLOT #
    ##############

    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1, aspect=4.0)
    ax1.patch.set_facecolor('white')

    pGlass = plt.contourf(x, y, temp, 20, vmin=np.min(temp0[np.nonzero(temp0)]),
                         vmax=np.max(temp0[np.nonzero(temp0)]),
                         cmap=cm.jet, alpha=0.5,linewidth=0.0, color='#444444',antialiased=True)
    cbar= plt.colorbar(pGlass, orientation='horizontal')
    cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')

    p = plt.contour(xg,yg,perm,[-14.15,-18],colors='black',linewidths=np.array([4.0]))
    #psiMask[np.abs(psiMask)>=1.0e-6] = 1.0e-6
    CS = plt.contour(xg, yg, psi, 8, colors='black',linewidths=np.array([1.0]))

##    plt.plot([x[xb], x[xb]], [-1200.0, 0.0],
##             linewidth=3.0, color=scolor)
##    plt.text(x[xb]-85.0, 20.0, '504B',
##        bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10},fontsize=12)
##
##    plt.plot([x[xa], x[xa]], [-1200.0, 0.0],
##             linewidth=3.0, color=scolor)
##    plt.text(x[xa]-85.0, 20.0, '896A',
##        bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10},fontsize=12)

    plt.savefig(path+'crThermal_'+str(i)+'.png')





print "ALL DONE!"




print "504B:" , 2.6*(temp[-1,xb] - temp[-2,xb])/(y[1]-y[0])
print "896A:" , 2.6*(temp[-1,xa] - temp[-2,xa])/(y[1]-y[0])



#######################
# HEAT FLOW BENCHMARK #
#######################


 
# ripped heat flow data from alt1996
ripX = np.array([0.36865786, 1.6313758, 2.531271, 2.8942623, 3.126651, 3.3882568,
        3.577378, 3.7666466, 3.9267824, 4.1012845, 4.2464643, 4.3480396,
        4.449467, 4.5653033, 4.6807604, 4.79626, 4.8972664, 4.9839487,
        5.1287913, 5.288379, 5.5207467, 5.767881, 6.0149517, 6.2473407,
        6.624719, 6.9003544, 7.3355355, 7.7417097, 8.089892, 8.27855,
        8.699429, 8.931648, 9.497811, 9.802664])
ripX = ripX*1000.0


ripQ = [222.5499, 223.25452, 224.79483, 230.41069, 236.84354, 250.53206,
        267.4522, 290.0175, 308.5529, 321.442, 323.04315, 322.22852, 315.76877,
        305.2756, 280.2664, 256.8701, 234.2814, 218.9519, 207.64995, 205.21774,
        210.84413, 226.14671, 239.02995, 245.4628, 246.23882, 241.3779, 232.47188,
        224.37463, 218.70143, 217.87976, 217.03937, 217.02065, 221.81367, 224.20844]

## ripX = ripX + 1500.0*np.ones(len(ripX))
ripXdata = np.array([0.71601903, 0.8902895, 0.992602, 1.5294846, 1.5292529, 1.6882722,
            2.1970122, 2.2840946, 2.2842631, 2.3851223, 3.1401324, 3.0970547,
            3.460699, 3.6640813, 3.910752, 3.867021, 3.9392319, 4.0263143, 4.14253,
            4.3016543, 4.3175583, 4.2025857, 4.289626, 4.576889, 4.7946796, 4.765442,
            4.6782117, 4.866933, 5.374746, 5.534271, 5.359937, 6.261686, 6.2178497,
            6.434966, 6.957883, 7.030599, 7.087369, 7.755402, 7.8712173, 8.582686,
            9.613668])
ripXdata = ripXdata*1000.0

## ripXdata = ripXdata + 1500.0*np.ones(len(ripXdata))
ripQdata = [185.42528, 189.44347, 216.85445, 211.97246, 203.10155, 178.8953, 207.88638,
            207.87936, 214.33093, 186.09712, 197.3265, 215.07184, 245.68753, 252.92915,
            250.48991, 243.23541,229.51997, 229.51295, 233.53583, 213.36179, 266.58606,
            310.14352, 308.52362, 193.17839, 196.38663, 188.32451, 182.6864, 184.28407,
            177.79155, 172.93999, 166.50247, 239.01006, 227.72333, 205.12527, 221.21207,
            226.85133, 177.65343, 192.92207, 181.62245, 192.85535, 212.12695]


fig=plt.figure()
ax1=fig.add_subplot(1,1,1)


hfTop = -1000.0*1.2*(temp[-1,:] - temp[-2,:])/(y[1]-y[0])
#ax1.plot(x,hfTop[::-1],'m-',linewidth=1.0, label='model')
print hfTop


ax1.plot(ripX,ripQ, 'r-', label='Slabel')
#ax1.scatter(ripXdata,ripQdata, 'b', label='label')


ax1.plot(x,hfTop,'g-',linewidth=2.0, label='model')

# plot data points
plt.scatter(ripXdata,ripQdata,10,color='k',label='ODP observations (Alt et al. 1996)',zorder=4)

plt.xlabel('LATERAL DISTANCE [m]')
plt.ylabel('HEAT FLOW [mW m$^{-2}$]')
plt.title('HEAT FLOW OUT OF RIFT FLANK')
plt.xlim([0,10000])
plt.ylim([0,400])





plt.savefig(path+'crFlux.png')



