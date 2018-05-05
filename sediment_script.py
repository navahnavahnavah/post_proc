# sediment_script.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import streamplot as sp
import multiplot_data as mpd
import heapq
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate



plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams['axes.titlesize'] = 11


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

x_grab = np.array([18.36022762301583, 19.169874365269266, 19.397983534391095, 19.134672332997376, 19.83167257198076, 20.993339636953067, 21.22567304994753, 22.46478458591799, 22.929451411906918, 24.23310000704251, 24.633229773866304, 25.38554749213408, 25.969146898584455, 26.33700813582569, 27.57611967179615, 28.040786497785078, 29.279898033755536, 29.744564859744457, 30.983676395714923, 31.448343221703844, 32.68745475767431, 33.15212158366323, 34.391233119633696, 34.85589994562262, 36.01756701059493, 36.559678307582004, 37.64390090155616, 38.2634566695414, 39.347679263515545, 39.96723503150078, 40.58679079948601, 41.67101339346017, 41.20634656747124, 42.29056916144539, 43.37479175541955, 43.994347523404784, 45.07857011737893, 45.69812588536416, 46.782348479338324, 47.01468189233279, 48.137626721806015, 48.486126841297704, 49.570349435271865, 50.1899052032571, 50.58819105410474, 50.49968308724971, 51.89368356521648, 51.89368356521648, 53.59746192717587, 53.59746192717587, 55.30124028913525, 55.38728970135543, 55.27542546546921, 57.005018651094645, 56.77268523810018, 58.27854995195317, 58.708797013054024, 58.399019129061415, 59.94790854902449, 60.41257537501342, 61.43041699384629, 62.1163537369728, 62.3680482677168, 63.82013209893219, 63.753751123790906, 65.52391046089157, 65.52391046089157, 67.22768882285096, 67.22768882285096, 68.93146718481034, 68.260281769493, 69.24124506880297, 70.63524554676974, 70.63524554676974, 71.79691261174204, 72.33902390872912, 72.98955746511362, 73.87242443449256, 74.0428022706885, 75.59169169065159, 75.74658063264789, 77.21802558161282, 77.45035899460727, 78.84435947257404, 79.15413735656666, 80.54813783453343, 80.85791571852604, 82.20028654916071, 82.56169408048544, 82.48424960948728, 83.87825008745405, 84.26547244244482, 85.50458397841528, 85.96925080440421, 87.20836234037468, 87.67302916636359, 88.44747387634513, 89.37680752832298, 89.63495576498349, 89.37680752832298, 91.08058589028236, 91.23547483227867, 92.78436425224176, 92.62947531024544, 94.02347578821221, 94.48814261420114, 95.9485240673092, 95.21095767685057, 96.19192097616053, 97.49556957129612, 97.89569933811991, 99.12289634009066, 99.5994777000793, 100.79986700055069, 101.30325606203868, 100.91603370704792, 101.6130339460313, 101.92281183002392, 102.2713119495156, 103.00703442399806, 102.77470101100361, 103.93636807597592, 104.71081278595746, 104.47847937296301, 105.09803514094823, 105.33036855394268, 106.41459114791685, 106.26938276479531, 107.28196922309617, 107.11159138690023, 107.11159138690023, 107.42136927089284])

y_grab = np.array([2517.2117335433913, 2589.3256507430237, 2612.3972181604076, 2696.5761516411367, 2747.697595080681, 2784.0100737948746, 2599.3501790025316, 2846.1366913826932, 2612.552606815796, 2811.7272138042663, 2623.7128725676093, 2786.280664366101, 2818.8649718305423, 2628.916832306856, 2795.718663946847, 2629.355747236051, 2780.5010030715825, 2628.4332207909442, 2767.6658646013443, 2629.5528564072893, 2744.790095995638, 2623.1845644649798, 2759.0136048396275, 2612.2214078844054, 2759.048890157465, 2610.1076202367854, 2724.1291681901293, 2627.7347325165274, 2728.1418667268636, 2633.7895931147123, 2765.1983353959895, 2664.690758793887, 2804.732475601535, 2842.286399698318, 2670.9157995638598, 2852.2554042476177, 2670.5038136341163, 2846.7380131642467, 2667.7093052993464, 2862.1200704914854, 2932.964507461849, 2667.4674995413907, 2921.1340341730147, 2665.3537118937707, 2881.715531347929, 2848.0058970868467, 2664.7715457922395, 2957.991981782271, 2665.0402805496465, 2938.236182992671, 2665.3090153070534, 2933.223995979181, 2988.689815007688, 2665.57775006446, 2904.218795347829, 2962.7997490026783, 2664.9955839629292, 2980.290581420365, 2922.6736273374972, 2662.371255799947, 2918.152835861197, 2662.6399905573535, 2905.6993389363397, 2661.2069235968843, 2946.191514431828, 2661.1352980107163, 2941.890036417306, 2658.851330191309, 2920.0778938852723, 2658.09898391799, 2965.622142578575, 3009.3997193525756, 2658.367718675397, 3014.0442777114768, 2985.913779874002, 2655.2328499970517, 3014.4156202853483, 2967.268611217648, 2653.28924252122, 3000.36073694926, 2653.557977278627, 2979.806112371596, 2652.9758111770957, 2980.9626959324332, 2650.862023529476, 3004.0355737093787, 2649.5991367407946, 2994.652621921324, 2647.6555292649628, 2968.177667610658, 3033.355313074408, 2644.5206605866174, 3049.23437238567, 2643.9384944850863, 3048.141665768777, 2644.207229242493, 3005.9962824879544, 2644.4759639999, 2978.5614496984645, 2914.790148867321, 2644.744698757307, 3015.081182999612, 2645.013433514714, 3015.3739912513734, 3048.0253435958903, 2644.771627756758, 3080.0262515392383, 3122.439561839683, 2654.9108124778454, 3099.5041360920086, 2664.8798170271452, 3050.6492873639563, 2671.955758656056, 3003.1502558617376, 2672.564853757038, 2952.3650701805145, 3073.8418329732485, 3121.0022380614314, 3157.5608533829773, 2672.8335885144447, 3210.9917361439866, 3247.338250892538, 2673.1023232718517, 3284.267764598273, 3207.6142288520323, 3145.875472141872, 2673.3710580292586, 3075.088839319814, 2955.8027374124786, 2987.9739524389515, 3029.1575540115505, 2670.7218830059583])


sed_top_index = np.where(y_grab<2690.0)
x_grab_sed_top = x_grab[sed_top_index]
y_grab_sed_top = y_grab[sed_top_index]

aq_top_index = np.where(y_grab>2690.0)
x_grab_aq_top = x_grab[aq_top_index]
y_grab_aq_top = y_grab[aq_top_index]

#todo: path
out_path = '../output/revival/local_fp_output/'

fig=plt.figure(figsize=(18.0,9.0))

ax=fig.add_subplot(2, 3, 1, frameon=True)
plt.scatter(x_grab,y_grab, facecolor='k', edgecolor='none')

y_line = 2690.0
plt.plot([0.0, 120.0],[y_line,y_line],lw=1)
plt.xlim([0.0,120.0])
plt.ylim([2400.0,3400.0])
plt.xlabel('distance from ridge axis [km]')


ax=fig.add_subplot(2, 3, 2, frameon=True)
plt.scatter(x_grab_sed_top,y_grab_sed_top, facecolor='b', edgecolor='none')
plt.scatter(x_grab_aq_top,y_grab_aq_top, facecolor='g', edgecolor='none')
y_line = 2690.0
plt.plot([0.0, 120.0],[y_line,y_line],lw=1)
plt.xlim([0.0,120.0])
plt.ylim([2400.0,3400.0])
plt.xlabel('distance from ridge axis [km]')


ax=fig.add_subplot(2, 3, 3, frameon=True)

y_line = 2690.0
plt.plot([0.0, 120.0],[y_line,y_line],lw=1)
plt.xlim([0.0,120.0])
plt.ylim([2400.0,3400.0])
plt.xlabel('distance from ridge axis [km]')



aq_top_y_interp = interpolate.interp1d(x_grab_aq_top, y_grab_aq_top)
sed_top_y_interp = interpolate.interp1d(x_grab_sed_top, y_grab_sed_top)
x_norm = np.linspace(np.min(x_grab_aq_top),np.max(x_grab_aq_top),100.0)
y_norm_aq_top = np.zeros(len(x_norm))
y_norm_sed_top = np.zeros(len(x_norm))
y_norm_aq_top = aq_top_y_interp(x_norm)
y_norm_sed_top = sed_top_y_interp(x_norm)

ax=fig.add_subplot(2, 3, 4, frameon=True)
plt.plot(x_norm,y_norm_aq_top, color='g')
plt.scatter(x_grab_aq_top,y_grab_aq_top, facecolor='g', edgecolor='none')

plt.plot(x_norm,y_norm_sed_top, color='b')
plt.scatter(x_grab_sed_top,y_grab_sed_top, facecolor='b', edgecolor='none')

y_line = 2690.0
plt.plot([0.0, 120.0],[y_line,y_line],lw=1)
plt.xlim([0.0,120.0])
plt.ylim([2400.0,3400.0])
plt.xlabel('distance from ridge axis [km]')




h_sed = np.zeros(len(x_norm))
h_sed = np.abs(y_norm_sed_top - y_norm_aq_top)
h_sed_conv1 = savitzky_golay(h_sed, 51, 3)
h_sed_conv2 = savitzky_golay(h_sed, 11, 3)
h_sed_conv3 = savitzky_golay(h_sed, 51, 7)
h_sed_conv4 = savitzky_golay(h_sed, 91, 7)

ax=fig.add_subplot(2, 3, 5, frameon=True)
plt.plot(x_norm,h_sed, color='#c1bfbf', lw=2, label='no smoothing')
plt.plot(x_norm,h_sed_conv1, color='r', label='smoothing 51, 3')
plt.plot(x_norm,h_sed_conv2, color='g', label='smoothing 11, 3')
plt.plot(x_norm,h_sed_conv3, color='m', label='smoothing 51, 7')
plt.plot(x_norm,h_sed_conv4, color='c', label='smoothing 91, 7')
plt.xlim([0.0,120.0])
plt.xlabel('distance from ridge axis [km]')
plt.title('sediment thickness [m]')

plt.legend(fontsize=8,loc='best')

plt.savefig(out_path+"data_grab.png",bbox_inches='tight')
