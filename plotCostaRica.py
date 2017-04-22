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
steps = 20

path = "output/marchCR/"
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


xb = 10
xa = 10

xcb = 10
xca = 10


# load output
psi0 = np.loadtxt(path + 'psiMat.txt')
perm0 = np.loadtxt(path + 'permMat.txt')
perm0 = np.log10(perm0)
temp0 = np.loadtxt(path + 'hMat.txt')
#temp0[temp0<273.0] = 273.0
temp0 = temp0 - 273.0
glass0 = np.loadtxt(path + 'pri_glass.txt')
pyrite_e0 = np.loadtxt(path + 'sec8.txt')
calcite_e0 = np.loadtxt(path + 'sec16.txt')
saponite_e0 = np.loadtxt(path + 'sec19.txt') #+ np.loadtxt(path + 'sec5.txt')
celadonite_e0 = np.loadtxt(path + 'sec6.txt')
zeolite_e0 = np.loadtxt(path+'sec20.txt')+np.loadtxt(path+'sec26.txt')#+ \
    #np.loadtxt(path+'sec26.txt')
fehydrox_e0 = np.loadtxt(path + 'sec10.txt')
quartz_e0 = np.loadtxt(path + 'sec17.txt')
anhydrite_e0 = np.loadtxt(path + 'sec14.txt')
chlorite_e0 = np.loadtxt(path + 'sec66.txt')
talc_e0 = np.loadtxt(path + 'sec49.txt')

ca_e0 = np.loadtxt(path + 'sol_ca.txt')
mg_e0 = np.loadtxt(path + 'sol_mg.txt')

mesh = np.loadtxt(path + 'mesh.txt')

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
    #
    # psiMask = np.ma.masked_where(mesh== 0, psi)
    # tempMask = np.ma.masked_where(mesh== 0, temp)

    glass = cut(glass0,i)
    pyrite_e = cut(pyrite_e0,i)
    celadonite_e = cut(celadonite_e0,i)
    calcite_e = cut(calcite_e0,i)
    saponite_e = cut(saponite_e0,i)
    zeolite_e = cut(zeolite_e0,i)
    fehydrox_e = cut(fehydrox_e0,i)
    quartz_e = cut(quartz_e0,i)
    anhydrite_e = cut(anhydrite_e0,i)
    chlorite_e = cut(chlorite_e0,i)
    talc_e = cut(talc_e0,i)

    ca_e = cut(ca_e0,i)
    mg_e = cut(mg_e0,i)


    # #############
   #  # FULL PLOT #
   #  #############
   #
   #  fig=plt.figure()
   #  ax1=fig.add_subplot(1,1,1, aspect='equal')
   #
   #  print xCell.shape
   #  print yCell.shape
   #  print celadonite_e.shape
   #
   #
   #  levs = 2
   #  alph = 0.6
   #
   #  celadonite_ef = celadonite_e
   #  #celadonite_ef[celadonite_ef<=np.max(celadonite_e)/5.0] = 0.0
   #  plotMin(celadonite_ef,celadonite_e0,'celadonite (stage 1)',cm.Greens,'green')
   #
   #  fehydrox_ef = fehydrox_e
   #  #fehydrox_ef[fehydrox_ef<=np.max(fehydrox_e)/2.0] = 0.0
   #  plotMin(fehydrox_ef,fehydrox_e0,'iron oxyhydroxide (stages 1 + 2)',cm.Reds,'red')
   #
   #  saponite_ef = saponite_e
   #  #saponite_ef[saponite_ef<=np.max(saponite_e)/2.0] = 0.0
   #  plotMin(saponite_ef,saponite_e0,'saponites (stages 2 + 3)',cm.Oranges,'Chocolate')
   #
   #  pyrite_ef = pyrite_e
   #  #pyrite_ef[pyrite_ef<=np.max(pyrite_e)/2.0] = 0.0
   #  plotMin(pyrite_ef,pyrite_e0,'pyrite (stage 3)',cm.Blues,'blue')
   #
   #  calcite_ef = calcite_e
   #  #calcite_ef[calcite_ef<=np.max(calcite_e)/10.0] = 0.0
   #  plotMin(calcite_ef,calcite_e0,'calcite (stages 3 + 4)',cm.Purples,'purple')
   #
   #  zeolite_ef = zeolite_e
   #  #zeolite_ef[zeolite_ef<=np.max(zeolite_e)/2.0] = 0.0
   #  plotMin(zeolite_ef,zeolite_e0,'total zeolites (stage 4)',cm.Greys,'black')
   #
   #  theLegend = plt.legend(prop={'size':10},loc=9, bbox_to_anchor=(1.22,1.03),ncol=1)
   #
   #  print xg.shape
   #  print yg.shape
   #  print psi.shape
   #
   #  p = plt.contour(xg,yg,perm,[-14.0,-18],colors='black',linewidths=np.array([4.0]))
   #  CS = plt.contour(xg, yg, psi, 20, colors='black',linewidths=np.array([1.0]))
   #
   #  plt.xlabel('x [m]')
   #  plt.ylabel('y [m]')
   #
   #  plt.title('' + str((i*.5)+.5) + 'Myr')
   #
   #  scolor = 'MediumSpringGreen'
   #
   #  plt.plot([x[xb], x[xb]], [-1200.0, 0.0],
   #           linewidth=3.0, color=scolor)
   #  plt.text(x[xb]-85.0, 20.0, '504B',
   #      bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10},fontsize=12)
   #
   #  plt.plot([x[xa], x[xa]], [-1200.0, 0.0],
   #           linewidth=3.0, color=scolor)
   #  plt.text(x[xa]-85.0, 20.0, '896A',
   #      bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10},fontsize=12)
   #
   #  plt.savefig(path+'j_final_'+str(i)+'.png',
   #              bbox_inches='tight',bbox_extra_artists=(theLegend,))


#
#
#     #############
#     # HEAT PLOT #
#     ##############
#
#     fig=plt.figure()
#     ax1=fig.add_subplot(1,1,1, aspect='equal')
#     ax1.patch.set_facecolor('white')
#
#     pGlass = plt.contourf(x, y, tempMask, 20, vmin=np.min(temp0[np.nonzero(temp0)]),
#                          vmax=np.max(temp0[np.nonzero(temp0)]/1.0),
#                          cmap=cm.jet, alpha=0.5,linewidth=0.0, color='#444444',antialiased=True)
#     cbar= plt.colorbar(pGlass, orientation='horizontal')
#     cbar.ax.set_xlabel('TEMPERATURE [$^{o}$C]')
#
#     p = plt.contour(xg,yg,perm,[-14.15,-18],colors='black',linewidths=np.array([4.0]))
#     CS = plt.contour(xg, yg, psiMask, 20, colors='black',linewidths=np.array([1.0]))
#
# ##    plt.plot([x[xb], x[xb]], [-1200.0, 0.0],
# ##             linewidth=3.0, color=scolor)
# ##    plt.text(x[xb]-85.0, 20.0, '504B',
# ##        bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10},fontsize=12)
# ##
# ##    plt.plot([x[xa], x[xa]], [-1200.0, 0.0],
# ##             linewidth=3.0, color=scolor)
# ##    plt.text(x[xa]-85.0, 20.0, '896A',
# ##        bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10},fontsize=12)
#
#     plt.savefig(path+'ccostaRicaTemp_'+str(i)+'.png')
#
#
# print "ALL DONE!"
#
#
#
#
# print "504B:" , 2.6*(temp[-1,xb] - temp[-2,xb])/(y[1]-y[0])
# print "896A:" , 2.6*(temp[-1,xa] - temp[-2,xa])/(y[1]-y[0])
#
#
#
# #######################
# # HEAT FLOW BENCHMARK #
# #######################
#
# # ripped heat flow data from alt1996
# ripX = np.array([0.0, 0.36865786, 1.6313758, 2.531271, 2.8942623, 3.126651, 3.3882568,
#         3.577378, 3.7666466, 3.9267824, 4.1012845, 4.2464643, 4.3480396,
#         4.449467, 4.5653033, 4.6807604, 4.79626, 4.8972664, 4.9839487,
#         5.1287913, 5.288379, 5.5207467, 5.767881, 6.0149517, 6.2473407,
#         6.624719, 6.9003544, 7.3355355, 7.7417097, 8.089892, 8.27855,
#         8.699429, 8.931648, 9.497811, 9.802664, 10.0])
# ripX = ripX*1000.0-1000.0
#
#
# ripQ = [-50.0, 222.5499, 223.25452, 224.79483, 230.41069, 236.84354, 250.53206,
#         267.4522, 290.0175, 308.5529, 321.442, 323.04315, 322.22852, 315.76877,
#         305.2756, 280.2664, 256.8701, 234.2814, 218.9519, 207.64995, 205.21774,
#         210.84413, 226.14671, 239.02995, 245.4628, 246.23882, 241.3779, 232.47188,
#         224.37463, 218.70143, 217.87976, 217.03937, 217.02065, 221.81367, 224.20844, -50.0]
#
# ## ripX = ripX + 1500.0*np.ones(len(ripX))
# ripXdata = np.array([0.71601903, 0.8902895, 0.992602, 1.5294846, 1.5292529, 1.6882722,
#             2.1970122, 2.2840946, 2.2842631, 2.3851223, 3.1401324, 3.0970547,
#             3.460699, 3.6640813, 3.910752, 3.867021, 3.9392319, 4.0263143, 4.14253,
#             4.3016543, 4.3175583, 4.2025857, 4.289626, 4.576889, 4.7946796, 4.765442,
#             4.6782117, 4.866933, 5.374746, 5.534271, 5.359937, 6.261686, 6.2178497,
#             6.434966, 6.957883, 7.030599, 7.087369, 7.755402, 7.8712173, 8.582686,
#             9.613668])
# ripXdata = ripXdata*1000.0-1000.0
#
# ## ripXdata = ripXdata + 1500.0*np.ones(len(ripXdata))
# ripQdata = [185.42528, 189.44347, 216.85445, 211.97246, 203.10155, 178.8953, 207.88638,
#             207.87936, 214.33093, 186.09712, 197.3265, 215.07184, 245.68753, 252.92915,
#             250.48991, 243.23541,229.51997, 229.51295, 233.53583, 213.36179, 266.58606,
#             310.14352, 308.52362, 193.17839, 196.38663, 188.32451, 182.6864, 184.28407,
#             177.79155, 172.93999, 166.50247, 239.01006, 227.72333, 205.12527, 221.21207,
#             226.85133, 177.65343, 192.92207, 181.62245, 192.85535, 212.12695]
#
# # bathymetry
# xBath0 = np.array([-0.011880637,
# 0.30742204, 0.670266, 0.9315768, 1.2220626, 1.3673266, 1.6141447,
# 1.8897797, 2.2088716, 2.5424564, 3.0066018, 3.3547845, 3.6450596,
#      3.8774905, 4.0519505, 4.31324, 4.443801, 4.5743194, 4.7338653,
#      4.8934746, 5.270811, 5.517608, 5.677407, 6.0262218, 6.345735,
#      6.6652694, 7.1152167, 7.507067, 7.78266, 8.174321, 8.580538,
#      8.9577265, 9.262326, 9.537835, 9.726282, 9.973016])*1000.0-1000.0
# bath0 = np.array([100.806946, 100.7812, 100.75194, 103.15021, 111.19126,
#         116.018234, 119.22412, 114.363205, 106.27299, 97.37516,
#         86.04746, 80.374245, 80.35084, 88.39657, 99.672775, 101.264595,
#         98.834724, 94.79196, 90.74686, 89.121086, 88.28421, 90.68365,
#         96.315895, 114.83607, 122.87478, 131.71994, 132.4901, 131.65205,
#         125.178246, 117.08218, 110.59784, 104.11584, 96.83324,87.133644,
#         78.24752, 78.22762])
#
# x3 = np.linspace(0.0,3000.0,40)
#
# hf3 = [  53.78498147,   56.17903532,   63.36119685,   76.72933643,  102.31411319,
#   162.70461236,  -19.89867572,   40.47754124,   67.58337234,   84.03468194,
#    95.95496346,  105.40085376,  113.24356706,  119.9954773,   126.07255358,
#   131.78721902,  137.3126453,   142.72916981,  148.04393364,  153.20159326,
#   158.10931438,  162.65819521,  166.78218057,  170.54732566,  174.25177139,
#   178.50072601,  184.1189866,   192.0759575,   203.46779761,  218.99254945,
#   238.37706596,  260.23954424,  281.92706556,  300.24577287,  313.6674707,
#   322.65633059,  328.10320477,  330.90787187,  332.06115956,  332.44499369]
#
#
#
#
# fig=plt.figure()
# ax1=fig.add_subplot(1,1,1, aspect=10.0)
#
# # plot observed heat flow
# #plt.plot(ripX,ripQ,'k:',linewidth=2.0,label='Maximum heat flow curve (Alt et al. 1996)')
#
# plt.fill(ripX, ripQ, facecolor='grey',alpha=0.2,
#          label='Below maximum heat flow curve (Alt et al. 1996)')
#
# # plot experiment heat flow
# ##plt.plot(x[2:-2],-1000.0*1.8*(temp[-2,2:-2] - temp[-3,2:-2])/(y[1]-y[0]),
# ##         linewidth=2.0,color='c')
# ##
#
#
#
# # begins
# hfRho18 = [133.46537757,  167.72123525,  299.96089792,  283.82558201,  255.06301565,
#   223.57790485,  198.07703601,  183.5448971,   179.24238422,  180.63489876,
#   182.96825325,  183.38243707,  179.92257402,  177.43491681,  174.61239695,
#   177.16534027,  182.94070506,  191.25651625,  201.90389666,  214.88463156,
#   230.34868405,  248.65667971,  270.65840944,  299.35492061,  323.7863108,
#   334.1194828,   329.26710521,  312.04991262,  279.32056399,  245.56815387,
#   220.08335251,  199.35630916,  182.08020249,  167.48200839,  155.01900326,
#   144.41804001,  136.22005697,  130.65447127,   84.38014194,   68.95715077]
#
# hfRho16 = [ 145.57311296,  181.96237434,  318.79340402,  298.35736051,  264.0033127,
#   226.34472076,  194.17049379,  172.76326065,  162.28905192,  159.26836653,
#   159.29871621,  159.15410893,  156.36193875,  151.91976577,  151.1306742,
#   155.5670827,   163.62045846,  174.99444581,  189.84079314,  208.48977632,
#   231.34665281,  258.89523213,  291.85140957,  332.96450873,  364.76660072,
#   377.8133908,   370.63658509,  347.29411379,  303.62172799,  257.91333085,
#   223.22900715,  195.66436036,  173.54837302,  155.66348755,  141.04387016,
#   129.04146599,  119.82766145,  114.17566256,   73.04471065,   59.33558353]
#
# hfK1 = [  69.34365355,   93.58768874,  206.72192949,  219.26705727,  229.10570808,
#   237.48578917,  244.42336802,  249.66672085,  253.00875575,  254.28879796,
#   253.2694059,   249.38286212,  240.98492827,  233.21226524,  221.97574406,
#   216.53244043,  213.64743595,  212.25135086,  211.92107497,  212.51735684,
#   214.09732526,  216.98590029,  222.11142497,  232.47137609,  245.55395887,
#   251.66852592,  251.26683903,  244.77736422,  231.03780022,  218.86222437,
#   211.57651689,  206.16534821,  201.69859003,  197.85667812,  194.6449683,
#   192.45622109,  192.84362577,  192.74107557,  129.4673572, 108.37611774]
#
# hfK2 = [  62.34537534,   87.18926296,  200.35799193,  220.23345907,  237.5202774,
#   250.78308575,  258.04736994,  259.06319144,  255.38195429,  249.0049518,
#   241.34612183,  232.75716356,  222.0419427,   211.103419,    201.21299522,
#   198.09054915,  198.51365934,  201.28976204,  205.87077782,  212.00141235,
#   219.6334632,   228.98116336, 240.86573938,  258.29180916,  276.33611116,
#   284.55194694,  283.04160425,  273.08869584,  252.59856702,  231.1609842,
#   215.22561909,  201.78071268,  189.96397712,  179.44870706,  170.14206822,
#   162.24222612,  156.85426601,  153.26962964,  100.95214425,   83.51179228]
#
# hfRho20 = [ 123.74634019,  156.01340146,  295.16038872,  281.85294849,  257.38396864,
#   231.54389754,  212.97525173,  205.77702273,  207.5605125,   212.60927035,
#   216.2333787,   216.34228048,  211.6684304,   199.0028247,   193.7434044,
#   194.03618951,  197.28360483,  202.54302513,  209.41633408,  217.76428022,
#   227.65827455,  239.45180154,  254.09819806,  274.67170775,  294.16087445,
#   302.64628677,  299.33460155,  286.271744,    260.96079536,  235.38494497,
#   216.45031778,  200.88807516,  187.57170855,  175.94242686,  165.68780697,
#   156.75964644,  149.94703684,  144.86055266,   94.33905398,   77.5003397 ]
#
#
# hfTop = -1000.0*1.8*(temp[-2,:] - temp[-3,:])/(y[1]-y[0])
#
# def corners(hf0):
#
#     hf = hf0
#
# ##    hf[:13] = hf[:13] - 1000.0*1.8*(temp[-2,13]-temp[-2,12])/(x[1]-x[0])
#     hf[24:28] = hf[24:28] + 1000.0*1.8*(temp[-2,24]-temp[-2,23])/(x[2]-x[0])
#     hf[24:28] = hf[24:28] - 1000.0*1.8*(temp[-2,28]-temp[-2,27])/(x[2]-x[0])
#
#     hf[2:13] = hf[2:13] + 1000.0*1.8*(temp[-2,2]-temp[-2,1])/(x[1]-x[0])
#     hf[2:13] = hf[2:13] - 1000.0*1.8*(temp[-2,13]-temp[-2,12])/(x[2]-x[0])
#
#     hf[37:] = hf[37:] + 1000.0*1.8*(temp[-2,37]-temp[-2,36])/(x[1]-x[0])
#
#     return hf
#
#
# # oh god why
#
# k5rho225 = [ 115.88755941,  142.11860574, 257.64917883, 242.27974591,  208.72550132,
#   170.50633302,  144.32527439,  142.42395646,  159.68756621,  181.98386632,
#   199.50098513,  207.80072884,  203.874909,    201.43436932,  191.90814165,
#   190.89410542,  193.85944728,  199.50627227,  207.32220644,  217.48934786,
#   231.29130928,  251.8558926,   285.12092263, 342.61570566,  378.24857144,
#   402.40869871,  392.6914466,   355.06320435,  305.50519317, 246.96423896,
#   212.40845506,  190.85482937,  176.91897232,  167.28205759,  159.9392213,
#   153.99425538,  151.00570495,  139.74645112,   92.9204718,    77.31002675]
#
# ax1.plot(x[2:-3],k5rho225[2:-3],linewidth=2.0, label='k5rho225')
#
# k5rho250 = [ 112.775825,    138.15350985,  251.95765211,  236.58643392,  203.76772265,
#   167.86762664,  146.82458961,  151.5555685,   173.25023092,  196.91048118,
#   214.17052038,  221.69723999,  216.85735933,  216.06469837,  205.06561878,
#   203.01433774,  204.87995017,  209.12711952,  215.08815291,  222.80768224,
#   233.37115472,  249.55824361,  276.81225254,  326.10905253,  356.94087707,
#   378.81942282,  369.98945242,  335.83714068,  292.18525601,  241.10496624,
#   212.03533258,  194.41645309,  183.14779697,  175.21760519,  168.91022844,
#   163.55440328,  160.94076061,  148.70610709,   99.03796988,  82.48311433]
#
# ax1.plot(x[2:-3],k5rho250[2:-3],linewidth=2.0, label='k5rho250')
#
# k3rho200 = [ 108.68575985,  134.37586784,  247.29793801,  237.17721374,  213.31522741,
#   188.94800823,  174.53548274,  173.41254473,  181.40700641,  192.05081627,
#   200.30235756 , 203.13201851,  197.66193576,  195.71256278,  188.25546889,
#   189.28735787,  194.05404226,  201.45222207,  211.24088518,  223.59141798,
#   239.18579556,  259.57363664,  287.87203148,  332.6163801,   353.96068146,
#   370.59944494,  363.34408713,  335.62590668,  300.7438646,   253.10915568,
#   223.18080472, 202.22167562,  187.05397878,  175.89065389,  167.6248304,
#   161.93158826,  160.97289556,  152.22017222,  103.62498532,   87.42361423]
#
# ax1.plot(x[2:-3],k3rho200[2:-3],linewidth=2.0, label='k3rho200')
#
# k3rho225 = [ 104.2672041,   129.15929414,  240.4122763,   231.86973518,  210.81955705,
#   189.96575917,  179.15234819,  181.09294217,  191.21009588,  203.02683145,
#   211.78360558,  214.74002106,  209.13604267,  209.4145488,   201.32903809,
#   201.96816656,  206.11714578,  212.43701946,  220.5136038,   230.36475153,
#   242.5028366,   258.20433068,  280.21141624,  316.56854266,  332.69290596,
#   346.67518016,  340.35887703,  316.60043665,  288.26300672,  248.36567987,
#   223.89491474,  206.67236632,  193.94514048,  184.31715213,  176.99930969,
#   171.93448437,  171.61313486,  162.27691581,  110.79493915,   93.63130481]
#
# ax1.plot(x[2:-3],k3rho225[2:-3],linewidth=2.0, label='k3rho225')
#
# k3rho250 = [  99.71118217,  123.60173291,  232.56392286,  225.23001295,  207.04520128,
#   190.43857275,  184.39006087,  190.41893472,  203.20150409,  216.24472362,
#   225.23536877,  227.86150838,  221.58805184,  223.64676151,  214.09732526,
#   213.55638692,  216.36462458,  221.00812499,  226.95844673,  234.14417881,
#   242.95808174,  254.4606089,   271.09401655,  300.49214082,  312.6574709,
#   324.67951309,  319.52720929,  299.45357662,  276.96935675,  243.69361506,
#   223.95918465,  210.08581222,  199.67051757,  191.60643016,  185.35618221,
#   181.07330736,  181.5356936,   171.80179007,  117.63297446,   99.57670259]
#
# ax1.plot(x[2:-3],k3rho250[2:-3],linewidth=2.0, label='k3rho250')
#
#
#
#
#
#
# ##
# ##ax1.plot(x[2:-3],hfRho16[2:-3],'r-',linewidth=2.0, label='rho1.6')
# ##
# ##ax1.plot(x[2:-3],hfRho20[2:-3],'m-',linewidth=2.0, label='rho2.0')
# ##
# ##ax1.plot(x[2:-3],hfK2[2:-3],'g-',linewidth=2.0, label='K 2e-14')
# ##
# ##ax1.plot(x[2:-3],hfK1[2:-3],'b-',linewidth=2.0, label='K 1e-14')
# ##
# ##ax1.plot(x[2:-3],hfRho18[2:-3],'c-',linewidth=2.0,label='K 3e-14')
# ##
# ax1.plot(x[2:-3],hfTop[2:-3],'k-',linewidth=2.0)
# hfTop = corners(hfTop)
# print hfTop
# ax1.plot(x[2:-3],hfTop[2:-3],'k-',linewidth=2.0, label='K 1e-14')
#
#
# ld = plt.legend(prop={'size':6},loc=9, bbox_to_anchor=(0.8,1.1),ncol=3)
#
#
# #plt.scatter(x[:],-1000.0*1.8*(temp[-2,:] - temp[-3,:])/(y[1]-y[0]), color='b')
# ##plt.plot(x[:],-1000.0*1.8*(temp[-1,:] - temp[-2,:])/(y[1]-y[0]),
# ##         linewidth=2.0,color='b',label='SWELERTIN model (Navah 2015)')
#
#
#
# # plot experiments 504b
# plt.plot([x[xb], x[xb]], [50.0, 300.0], linewidth=3.0, color=scolor)
# plt.text(x[xb]-170.0, 305.0, '504B',
#         bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10}, fontsize=12)
#
# # plot experiment 896a
# plt.plot([x[xa], x[xa]], [50.0, 335.0], linewidth=3.0, color=scolor)
# plt.text(x[xa]-170.0, 339.0, '896A',
#         bbox={'facecolor':scolor, 'alpha':1.0, 'pad':10}, fontsize=12)
#
# # plot data points
# plt.scatter(ripXdata,ripQdata,color='k',label='ODP observations (Alt et al. 1996)')
#
# plt.xlabel('LATERAL DISTANCE [m]')
# plt.ylabel('HEAT FLOW [mW m$^{-2}$]')
# plt.title('HEAT FLOW OUT OF RIFT FLANK')
# plt.xlim([0,5000])
# plt.ylim([0,375])
#
#
# ax2 = ax1.twinx()
#
# ax2.plot(xBath0,6.0*bath0-800.0,linewidth=2.0,color='r',label='ab')
#
#
# approx = -30.0*np.ones(40)
# approx[:12] = 0.0
# approx[24:29] = 0.0
#
# ax2.plot(np.linspace(0,5000.0,40),approx,linewidth=2.0,color='g')
# ax2.plot(xBath0,2.0*bath0-220.0,linewidth=2.0,color='r',label='ab')
#
# ##plt.plot(x[31:],-1000.0*1.8*(temp[-2,31:] - temp[-3,31:])/(y[1]-y[0]),
# ##         linewidth=2.0,color='c',label='SWELERTIN model (Navah 2015)')
# #plt.plot(x7,hf7,
# #         linewidth=2.0,color='red',label='Recharge zone 6km from discharge zone')
#
# ax2.set_ylabel('DEPTH [meters below seafloor]', color='r')
# for tl in ax2.get_yticklabels():
#     tl.set_color('r')
#
# plt.xlim([0,5000])
# plt.ylim([-400.0,1000])
#
# #ld = plt.legend(prop={'size':8},loc='upper left')
#
#
# plt.savefig(path+'costaRicaHeatFlow.pdf',bbox_extra_artists=(ld,))



###########################
# CROSS SECTION BENCHMARK #
###########################

fig=plt.figure()




ax1=fig.add_subplot(1,2,1,aspect=.02)


# observed 504b

# fe-hydrox
fehydrox_o = [-278.1507, -586.9178]

# aragonite
aragonite_o = [-276.57535, -594.7945]

# celadonite
celadonite_o = [-278.1507, -534.9315]

# phillipsite
phil_o = [-278.1507, -541.23285]

# saponite
saponite_o = [-276.57535, -810.61646]

# ML smectite chl
mlsc_o = np.array([-516.0274, -530.2055, None, -568.0137, -583.76715, None,
          -657.8082, -837.3973])

# ML chl-smect
mlcs_o = np.array([-788.56165, -837.3973])

# talc
talc_o = np.array([-508.1507, -522.3288, None, -547.53424, -599.52057, None,
          -670.41095, -681.43835, None, -728.6986, -736.5753, None,
          -758.6301, -768.0822, None, -831.0959, -842.1233])

# na-zeolite
nazeolite_o = np.array([-527.0548, -561.71234, None, -727.1233, -746.0274])

# pyrite
pyrite_o = np.array([-353.76712, -371.0959, None, -615.274, -635.7534, None,
            -711.3699, -840.548])

# anhydrite
anhydrite_o = np.array([-560.13696, -572.73975, None, -627.8767, -638.9041, None,
               -667.26025, -675.13696, None, -727.1233, -736.5753, None,
               -749.1781, -760.2055, None, -771.23285, -810.61646, None,
               -832.6712, -843.6986])

# calcite
calcite_o = np.array([-275.0, -314.38358, None, -377.39725, -393.1507, None,
             -405.75342, -440.41095, None, -451.43835, -465.61642, None,
             -520.7534, -566.43835, None, -632.6027, -670.41095, None,
             -694.0411, -706.64386, None, -747.6027, -755.47943, None,
             -769.65753, -777.53424, None, -823.2192, -838.9726])

# quartz
quartz_o = np.array([-638.9041, -649.9315, None, -671.9863, -692.46576, None,
            -736.5753, -816.9178])



# iron oxyhydroxide
fehydrox_e[:,xcb+1] = np.sum(fehydrox_e, axis=1)
fehydrox_e[fehydrox_e>0.0] = 1.0
fehydrox_e[fehydrox_e==0.0] = None
print fehydrox_e[:,xcb].shape
print yCell.shape
plt.plot(fehydrox_e[:,xcb],yCell,'k-')
plt.plot(fehydrox_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(1.2*np.ones(len(fehydrox_o)),fehydrox_o, 'r')
plt.text(1.0, -265.0, 'Fe-hydroxide', rotation='vertical',fontsize=8, ha='left', va='bottom')

# celadonite
celadonite_e[:,xcb+1] = np.sum(celadonite_e, axis=1)
celadonite_e[celadonite_e>0.0] = 2.0
celadonite_e[celadonite_e==0.0] = None
plt.plot(celadonite_e[:,xcb],yCell,'k-')
plt.plot(celadonite_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(2.2*np.ones(len(celadonite_o)),celadonite_o, 'r')
plt.text(2.0, -265.0, 'Celadonite', rotation='vertical',fontsize=8, ha='left', va='bottom')

# zeolites
zeolite_e[:,xcb+1] = np.sum(zeolite_e, axis=1)
zeolite_e[zeolite_e>0.0] = 3.0
zeolite_e[zeolite_e==0.0] = None
plt.plot(zeolite_e[:,xcb],yCell,'k-')
plt.plot(zeolite_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(3.2*np.ones(len(nazeolite_o)),nazeolite_o, 'r')
plt.plot(3.2*np.ones(len(phil_o)),phil_o, 'r')
plt.text(3.0, -265.0, 'Zeolites', rotation='vertical',fontsize=8, ha='left', va='bottom')

# saponite
saponite_e[:,xcb+1] = np.sum(saponite_e, axis=1)
saponite_e[saponite_e>0.0] = 4.0
saponite_e[saponite_e==0.0] = None
plt.plot(saponite_e[:,xcb],yCell,'k-')
plt.plot(saponite_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(4.2*np.ones(len(saponite_o)),saponite_o, 'r')
plt.text(4.0, -265.0, 'Saponite', rotation='vertical',fontsize=8, ha='left', va='bottom')

# pyrite?
pyrite_e[:,xcb+1] = np.sum(pyrite_e, axis=1)
pyrite_e[pyrite_e>0.0] = 5.0
pyrite_e[pyrite_e==0.0] = None
plt.plot(pyrite_e[:,xcb],yCell,'k-')
plt.plot(pyrite_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(5.2*np.ones(len(pyrite_o)),pyrite_o, 'r')
plt.text(5.0, -265.0, 'Pyrite', rotation='vertical',fontsize=8, ha='left', va='bottom')

# calcite
calcite_e[:,xcb+1] = np.sum(calcite_e, axis=1)
calcite_e[calcite_e>0.0] = 6.0
calcite_e[calcite_e==0.0] = None
plt.plot(calcite_e[:,xcb],yCell,'k-')
plt.plot(calcite_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(6.2*np.ones(len(calcite_o)),calcite_o, 'r')
plt.text(6.0, -265.0, 'Calcite', rotation='vertical',fontsize=8, ha='left', va='bottom')

# quartz
quartz_e[:,xcb+1] = np.sum(quartz_e, axis=1)
quartz_e[quartz_e>0.0] = 7.0
quartz_e[quartz_e==0.0] = None
plt.plot(quartz_e[:,xcb],yCell,'k-')
plt.plot(quartz_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(7.2*np.ones(len(quartz_o)),quartz_o, 'r')
plt.text(7.0, -265.0, 'Quartz', rotation='vertical',fontsize=8, ha='left', va='bottom')

# anhydrite
anhydrite_e[:,xcb+1] = np.sum(anhydrite_e, axis=1)
anhydrite_e[anhydrite_e>0.0] = 8.0
anhydrite_e[anhydrite_e==0.0] = None
plt.plot(anhydrite_e[:,xcb],yCell,'k-')
plt.plot(anhydrite_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(8.2*np.ones(len(anhydrite_o)),anhydrite_o, 'r')
plt.text(8.0, -265.0, 'Anhydrite', rotation='vertical',fontsize=8, ha='left', va='bottom')

# talc
talc_e[:,xcb+1] = np.sum(talc_e, axis=1)
talc_e[talc_e>0.0] = 9.0
talc_e[talc_e==0.0] = None
plt.plot(talc_e[:,xcb],yCell,'k-')
plt.plot(talc_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(9.2*np.ones(len(talc_o)),talc_o, 'r')
plt.text(9.0, -265.0, 'Talc', rotation='vertical',fontsize=8, ha='left', va='bottom')

# aragonite
plt.plot(10.2*np.ones(len(aragonite_o)),aragonite_o, 'r')
plt.text(10.0, -265.0, 'Aragonite', rotation='vertical',fontsize=8, ha='left', va='bottom')


plt.ylim([-850.0,-275.0])
plt.xlim([0,11])
plt.xticks([])
plt.text(3.0, -900.0, 'HOLE 504B', ha='center')
plt.ylabel('depth [m]')









# observed 896a

ax2=fig.add_subplot(1,2,2,aspect=.042)

# saponite
saponite_o = [-200.72995, -425.73584]

# chlorite
chlorite_o = [-230.21368, -239.24689, None, -286.8765, -295.9097, None,
              -317.2609, -331.2213, None, -349.29065, -364.89346, None,
              -370.64185, -378.85092, None, -388.70532, -396.0961]

# fe hydroxide

fehydrox_o = [-201.38927,-220.27098, None, -227.66179,-294.179, None,
              -300.7486,-334.41782, None,-345.0934,-395.18365, None,
              -411.60767,-423.10446]

# celadonite

celadonite_o = [-202.12216, -294.9178, None, -302.3086, -308.05698, None,
                -316.26898, -334.3354, None, -341.7262, -356.50778, None,
                -374.5742, -396.74658,None, -409.8858, -419.7402]

# carbonate
carbonate_o = [-201.22444, -234.07245, None, -239.82085, -298.94727, None,
               -303.05325, -342.46793, None, -347.3951, -411.44577, None,
               -418.83658, -425.4062]

# aragonite
aragonite_o = [-206.06923, -236.45363, None, -242.20203, -298.86484, None,
               -307.89804, -315.28885, None, -321.85843, -341.56726, None,
               -346.49445, -381.49445, None, -387.5574, -409.7298, None,
               -417.9418, -423.6902]

# calcite
calcite_o = [-201.88376, -207.63216, None, -214.20175, -220.77136, None,
             -228.98335, -234.73177, None, -275.79178, -293.03696, None,
             -299.60657, -324.24258, None, -345.5967 -353.80872, None,
             -360.37833, -385.01138]

# phil
phil_o = [-213.2952, -219.0436, None, -226.4344, -234.64641, None,
          -245.322, -255.1764, None, -276.52762, -283.0972, None,
          -299.5212,-308.5544, None, -321.6936,-332.3692]

# zeolite
zeolite_o = [-226.35493, -233.74573, None,-401.2676, -410.3008]



# iron oxyhydroxide
#fehydrox_e[:,xca+1] = np.sum(fehydrox_e, axis=1)
fehydrox_e[fehydrox_e>0.0] = 1.0
fehydrox_e[fehydrox_e==0.0] = None
plt.plot(fehydrox_e[:,xcb],yCell,'k-')
plt.plot(fehydrox_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(1.2*np.ones(len(fehydrox_o)),fehydrox_o, 'r')
plt.text(1.0, -195.0, 'Fe-hydroxide', rotation='vertical',fontsize=8, ha='left', va='bottom')

# celadonite
#celadonite_e[:,xca+1] = np.sum(celadonite_e, axis=1)
celadonite_e[celadonite_e>0.0] = 2.0
celadonite_e[celadonite_e==0.0] = None
plt.plot(celadonite_e[:,xcb],yCell,'k-')
plt.plot(celadonite_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(2.2*np.ones(len(celadonite_o)),celadonite_o, 'r')
plt.text(2.0, -195.0, 'Celadonite', rotation='vertical',fontsize=8, ha='left', va='bottom')

# zeolites
#zeolite_e[:,xca+1] = np.sum(zeolite_e, axis=1)
zeolite_e[zeolite_e>0.0] = 3.0
zeolite_e[zeolite_e==0.0] = None
plt.plot(zeolite_e[:,xcb],yCell,'k-')
plt.plot(zeolite_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(3.2*np.ones(len(zeolite_o)),zeolite_o, 'r')
plt.plot(3.2*np.ones(len(phil_o)),phil_o, 'r')
plt.text(3.0, -195.0, 'Zeolites', rotation='vertical',fontsize=8, ha='left', va='bottom')

# saponite
#saponite_e[:,xca+1] = np.sum(saponite_e, axis=1)
saponite_e[saponite_e>0.0] = 4.0
saponite_e[saponite_e==0.0] = None
plt.plot(saponite_e[:,xcb],yCell,'k-')
plt.plot(saponite_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(4.2*np.ones(len(saponite_o)),saponite_o, 'r')
plt.text(4.0, -195.0, 'Saponite', rotation='vertical',fontsize=8, ha='left', va='bottom')

# calcite
#calcite_e[:,xca+1] = np.sum(calcite_e, axis=1)
calcite_e[calcite_e>0.0] = 5.0
calcite_e[calcite_e==0.0] = None
plt.plot(calcite_e[:,xcb],yCell,'k-')
plt.plot(calcite_e[:,xcb+1]-.2,yCell,'b-')
plt.plot(5.2*np.ones(len(calcite_o)),calcite_o, 'r')
plt.text(5.0, -195.0, 'Calcite', rotation='vertical',fontsize=8, ha='left', va='bottom')

# chlorite
chlorite_e[:,xca+1] = np.sum(chlorite_e, axis=1)
chlorite_e[chlorite_e>0.0] = 6.0
chlorite_e[chlorite_e==0.0] = None
plt.plot(chlorite_e[:,xca],yCell,'k-',label='SWELERTIN model hole')
plt.plot(chlorite_e[:,xca+1]-.2,yCell,'b-',label='SWELERTIN model volcanic section')
plt.plot(6.2*np.ones(len(chlorite_o)),chlorite_o, 'r',label='ODP observations (Alt et al. 1996)')
plt.text(6.0, -195.0, 'Chlorite', rotation='vertical',fontsize=8, ha='left', va='bottom')


# aragonite
plt.plot(7.2*np.ones(len(aragonite_o)),aragonite_o, 'r')
plt.text(7.0, -195.0, 'Aragonite', rotation='vertical',fontsize=8, ha='left', va='bottom')



# carbonate
plt.plot(8.2*np.ones(len(carbonate_o)),carbonate_o, 'r')
plt.text(8.0, -195.0, 'Carbonate', rotation='vertical',fontsize=8, ha='left', va='bottom')

plt.ylim([-425.0,-200.0])
plt.ylabel('depth [m]')
plt.xlim([0,9])
plt.xticks([])
plt.text(3.0, -550.0, 'HOLE 896A', ha='center')

lgd = plt.legend(prop={'size':8},loc=9, bbox_to_anchor=(0.6,-.2),ncol=1)

plt.savefig(path+'costaRicaMineralsSums.eps',bbox_extra_artists=(lgd,))



#####################
# VERTICAL PROFILES #
#####################


fig=plt.figure()

ax1=fig.add_subplot(1,2,1)

for i in range(bitsC):
    plt.plot(ca_e[bitsC/2:,i],yCell[bitsC/2:],'r--',label='ca' if i == 0 else "")
    plt.plot(mg_e[bitsC/2:,i],yCell[bitsC/2:],'b--',label='mg' if i == 0 else "")
plt.legend()
plt.xlim([0,.1])
plt.ylim([-1000,0])


##ax1=fig.add_subplot(1,2,2)
##
##plt.plot(ca_e[bitsC/2:,xcb],yCell[bitsC/2:],'r--')
##plt.plot(mg_e[bitsC/2:,xcb],yCell[bitsC/2:],'b--')



plt.savefig(path+'costaRicaVertical.png')

