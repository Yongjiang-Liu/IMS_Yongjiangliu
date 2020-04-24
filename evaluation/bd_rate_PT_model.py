# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import scipy

def pchip_slopes(h, delta):
    d = np.zeros(len(h) + 1)
    k = np.argwhere(np.sign(delta[:-1]) * np.sign(delta[1:]) > 0).reshape(-1) + 1
    w1 = 2*h[k] + h[k-1]
    w2 = h[k] + 2*h[k-1]
    d[k] = (w1 + w2) / (w1 / delta[k-1] + w2 / delta[k])
    d[0] = pchip_end(h[0], h[1], delta[0], delta[1])
    d[-1] = pchip_end(h[-1], h[-2], delta[-1], delta[-2])
    return d

def pchip_end(h1, h2, del1, del2):
    d = ((2*h1 + h2)*del1 - h1*del2) / (h1 + h2)
    if np.sign(d) != np.sign(del1):
        d = 0
    elif np.sign(del1) != np.sign(del2) and np.abs(d) > np.abs(3*del1):
        d = 3 * del1
    return d

def spline_slopes(h, delta):
    a, r = np.zeros([3, len(h)+1]), np.zeros(len(h)+1)
    a[0, 1], a[0, 2:] = h[0] + h[1], h[:-1]
    a[1, 0], a[1, 1:-1], a[1, -1] = h[1], 2*(h[:-1] + h[1:]), h[-2]
    a[2, :-2], a[2, -2] = h[1:], h[-1] + h[-2]
    
    r[0] = ((h[0] + 2*a[0, 1])*h[1]*delta[0] + h[0]**2*delta[1]) / a[0, 1]
    r[1:-1] = 3*(h[1:] * delta[:-1] + h[:-1] * delta[1:])
    r[-1] = (h[-1]**2*delta[-2] + (2*a[2, -2] + h[-1])*h[-2]*delta[-1]) / a[2, -2]
    
    d = scipy.linalg.solve_banded((1, 1), a, r)
    return d

class PCHIP:
    def __init__(self, x, y, use_spline=False):
        assert len(np.unique(x)) == len(x)
        order = np.argsort(x)
        self.xi, self.yi = x[order], y[order]
        
        h = np.diff(self.xi)
        delta = np.diff(self.yi) / h
        
        self.d = spline_slopes(h, delta) if use_spline else pchip_slopes(h, delta)
        self.c = (3*delta - 2*self.d[:-1] - self.d[1:]) / h
        self.b = (self.d[:-1] - 2*delta + self.d[1:]) / h**2
        
        """
        The piecewise function is like p(x) = y_k + s*d_k + s*s*c_k + s*s*s*b_k
        where s = x - xi_k, k is the interval includeing x.
        So the original function of p(x) is P(x) = xi_k*y_k + s*y_k + 1/2*s*s*d_k + 1/3*s*s*s*c_k + 1/4*s*s*s*s*b_k + C.
        """
        self.interval_int_coeff = []
        self.interval_int = np.zeros(len(x)-1)
        for i in range(len(x)-1):
            self.interval_int_coeff.append(np.polyint([self.b[i], self.c[i], self.d[i], self.yi[i]]))
            self.interval_int[i] = np.polyval(self.interval_int_coeff[-1], h[i]) - np.polyval(self.interval_int_coeff[-1], 0)
    
    def interp(self, x):
        if len(x[x < np.min(self.xi)]) > 0 or len(x[x > np.max(self.xi)]) > 0:
            print('Warning: Some samples are out of the interval and the results may be strange!')
        
        """find the intervals the xs belong to"""
        k = np.zeros(len(x), dtype='int')
        for i in range(1, len(self.xi)-1):
            idx = np.argwhere(self.xi[i] <= x).reshape(-1)
            k[idx] = i * np.ones(len(idx), dtype='int')
        s = x - self.xi[k]
        y = self.yi[k] + s*(self.d[k] + s*(self.c[k] + s*self.b[k]))
        return y
    
    def _integral(self, lower, upper):
        assert lower <= upper
        if lower < np.min(self.xi):
            lower = np.min(self.xi)
            print('Warning: The lower bound is less than the interval and clipped!')
        elif lower > np.max(self.xi):
            print('Warning: The lower bound is greater than the interval!')
            return 0
        if upper > np.max(self.xi):
            upper = np.max(self.xi)
            print('Warning: The upper bound is greater than the interval and clipped!')
        elif upper < np.min(self.xi):
            print('Warning: The lower bound is less than the interval!')
            return 0
        left = np.arange(len(self.xi))[self.xi - lower > -1e-6][0]
        right = np.arange(len(self.xi))[self.xi - upper < 1e-6][-1]
        
        inte = np.sum(self.interval_int[left:right])
        if self.xi[left] - lower > 1e-6:
            inte += (np.polyval(self.interval_int_coeff[left-1], self.xi[left]-self.xi[left-1]) - np.polyval(self.interval_int_coeff[left-1], lower-self.xi[left-1]))
        if self.xi[right] - upper < -1e-6:
            inte += (np.polyval(self.interval_int_coeff[right], upper-self.xi[right]) - np.polyval(self.interval_int_coeff[right], 0))
        return inte
    
    def integral(self, lower, upper):
        if lower > upper:
            return -self._integral(upper, lower)
        else:
            return self._integral(lower, upper)

def BD_Rate(R1, PSNR1, R2, PSNR2, piecewise=True):
    lR1, lR2 = np.log10(R1), np.log10(R2)
    
    min_int = np.max((np.min(PSNR1), np.min(PSNR2)))
    max_int = np.min((np.max(PSNR1), np.max(PSNR2)))

    if piecewise == True:
        int1 = PCHIP(PSNR1, lR1, use_spline=False).integral(min_int, max_int)
        int2 = PCHIP(PSNR2, lR2, use_spline=False).integral(min_int, max_int)
    else:
        p1 = np.polyfit(PSNR1, lR1, len(lR1)-1)
        p2 = np.polyfit(PSNR2, lR2, len(lR1)-1)
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.power(10, avg_exp_diff) - 1) * 100.
    return avg_diff

def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=True):
    lR1, lR2 = np.log10(R1), np.log10(R2)
    
    min_int = np.max((np.min(lR1), np.min(lR2)))
    max_int = np.min((np.max(lR1), np.max(lR2)))

    if piecewise == True:
        int1 = PCHIP(lR1, PSNR1, use_spline=False).integral(min_int, max_int)
        int2 = PCHIP(lR2, PSNR2, use_spline=False).integral(min_int, max_int)
    else:
        p1 = np.polyfit(lR1, PSNR1, len(lR1)-1)
        p2 = np.polyfit(lR2, PSNR2, len(lR1)-1)
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    
    avg_diff = (int2 - int1) / (max_int - min_int)
    return avg_diff
    # CSGO_data:
    # [15,19,23,27,31,35,39,43]
    # [14,21,28,35,42,49,56,63]
    # bitrate_x264:[44717737,28556558,17876520,11047111,6874432,4306235,2721324,1729432]
    # bitrate_x265:[27159788,17007813,10328725,6165480,3664848,2170452,1261668,714638]
    # bitrate_vp9:[42282400,27869867,17241310,9519712,4932790,2770052,1610241,793034]
    # VMAF_x264:[98.23239,97.45365,95.75336,91.69574,84.24234,73.92583,60.78551,44.85119]
    # VMAF_x265:[97.39738,95.55946,91.87456,85.61612,76.59270,65.01197,51.07909,33.77761]
    # VMAF_vp9:[98.39090,98.11019,97.47503,95.85579,91.97011,85.66705,77.05557,63.38902]
    # PSNR_x264:[47.41830,45.23360,42.94486,40.41426,37.78265,35.18297,32.68521,30.31599]
    # PSNR_x265:[46.31816,44.11381,41.74416,39.29164,36.79149,34.29706,31.81427,29.46247]
    # PSNR_vp9:[48.80447,47.23509,45.42926,43.21305,40.72402,38.39043,36.11563,33.67343]
    # SSIM_x264:[0.99850,0.99723,0.99479,0.98011,0.98977,0.96293,0.93439,0.89032]
    # SSIM_x265:[0.99745,0.99508,0.99047,0.98209,0.96806,0.94589,0.91249,0.86534]
    # SSIM_vp9:[0.99896,0.99836,0.99728,0.99493,0.99011,0.98249,0.97037,0.94615]
    # MS_SSIM_x264:[0.99731,0.99531,0.99181,0.98555,0.97500,0.95792,0.93110,0.89119]
    # MS_SSIM_x265:[0.99558,0.99221,0.98643,0.97712,0.96293,0.94197,0.91166,0.86903]
    # MS_SSIM_vp9:[0.99807,0.99704,0.99528,0.99187,0.98568,0.97695,0.96434,0.94122]
    ########################################################################
    #DOTA2_data:
    #[15,19,23,27,31,35,39,43]
    #[14,21,28,35,42,49,56,63]
    #bitrate_x264:[57786622,36791469,23292084,14592875,9221104,5860761,3701256,2363690]
    #bitrate_x265:[35548730,22438025,13909012,8667587,5301400,3253005,1949843,1057675]
    #bitrate_vp9:[38272480,22407423,12547572,6377781,3146733,1769737,1036176,527080]
    # VMAF_x264:[98.17269,96.35334,93.24940,87.70294,79.37219,67.87735,53.62711,39.14197]
    # VMAF_x265:[96.27566,93.56716,89.28382,82.64782,72.68212,59.68019,45.62632,32.22301]
    # VMAF_vp9:[98.86659,98.12052,96.74876,94.14226,89.37910,82.72767,73.43238,59.90507]
    # PSNR_x264:[43.92663,41.66344,39.53302,37.39382,35.30803,33.26770,31.15330,29.18054]
    # PSNR_x265:[42.36156,40.27622,38.19695,36.18570,34.16182,32.17520,30.23038,28.35665]
    # PSNR_vp9:[45.35108,43.60645,41.85906,39.91160,37.86648,35.97382,34.07349,31.95399]
    # SSIM_x264:[0.99764,0.99564,0.99193,0.98463,0.97129,0.94907,0.91510,0.87293]
    # SSIM_x265:[0.99673,0.99402,0.98848,0.97848,0.96143,0.93484,0.89887,0.85626]
    # SSIM_vp9:[0.99835,0.99736,0.99570,0.99230,0.98542,0.97445,0.95760,0.93045]
    # MS_SSIM_x264:[0.99565,0.99253,0.98735,0.97848,0.96446,0.94332,0.91250,0.87462]
    # MS_SSIM_x265:[0.99408,0.98992,0.98260,0.97132,0.95423,0.92981,0.89789,0.85968]
    # MS_SSIM_vp9:[0.99686,0.99526,0.99274,0.98809,0.97982,0.96810,0.95154,0.92649]

#############################################
    #RUST_data:
    #[15,19,23,27,31,35,39,43]
    #[14,21,28,35,42,49,56,63]
    #bitrate_x264:[80504844,48141833,26722989,13480323,6567073,3303760,1821631,1149518]
    #bitrate_x265:[42703044,24487093,12492206,5740441,2684828,1332862,694992,381518]
    #bitrate_vp9:[94567412,62349752,36673831,18717696,8180933,3753516,1614600,556616]
    # VMAF_x264:[98.27559,96.14482,92.09401,84.57765,72.88599,59.41820,46.88850,33.01166]
    # VMAF_x265:[94.91047,89.99958,82.05893,71.18485,58.99906,46.44052,33.04755,19.18425]
    # VMAF_vp9:[98.88886,97.98665,96.16426,92.47454,85.46686,76.64064,65.44567,50.69008]
    # PSNR_x264:[42.20611,39.42623,37.02791,34.95988,33.24852,31.89563,30.82813,29.86827]
    # PSNR_x265:[39.33379,36.82175,34.72709,33.11893,31.92515,30.95143,30.09297,29.28370]
    # PSNR_vp9:[44.04660,41.69580,39.27806,36.95684,34.88696,33.48116,32.33472,31.19154]
    # SSIM_x264:[0.99683,0.99381,0.98769,0.97506,0.95134,0.92413,0.90351,0.88487]
    # SSIM_x265:[0.99425,0.98785,0.97542,0.95731,0.93615,0.91513,0.894740.87338]
    # SSIM_vp9:[0.99793,0.99651,0.99378,0.98813,0.97653,0.96168,0.94351,0.91974]
    # MS_SSIM_x264:[0.99415,0.98891,0.97925,0.96228,0.93689,0.91149,0.89272,0.87610]
    # MS_SSIM_x265:[0.98875,0.97817,0.96167,0.94172,0.92127,0.90226,0.88424,0.86506]
    # MS_SSIM_vp9:[0.99613,0.99361,0.98895,0.97997,0.96373,0.94640,0.92823,0.90690]
################################################################
    #STARCRAFT_data:
    #[15,19,23,27,31,35,39,43]
    #[14,21,28,35,42,49,56,63]
    #bitrate_x264:[50801933,35704080,24528516,16185521,10517980,6653432,4147998,2618464]
    #bitrate_x265:[33858017,23248360,15562324,10247805,6454984,3961004,2336288,1265061]
    #bitrate_vp9:[24021720,16545801,10823927,6247951,3268786,1829648,1056524,533615]
    # VMAF_x264:[97.83438,97.15630,95.62102,91.94054,85.81665,76.44454,63.33442,47.66344]
    # VMAF_x265:[97.36929,96.03336,93.35903,88.70782,81.08928,69.88626,55.45479,39.33598]
    # VMAF_vp9:[98.05202,97.79393,97.31328,96.11436,93.23703,88.71250,81.51719,70.25056]
    # PSNR_x264:[45.85571,43.17847,40.49352,37.73023,35.16151,32.75126,30.43580,28.27335]
    # PSNR_x265:[44.29027,41.66617,39.08879,36.59653,34.11851,31.72150,29.43679,27.33259]
    # PSNR_vp9:[48.03328,46.21320,44.08941,41.55335,38.87559,36.42115,34.03632,31.50169]
    # SSIM_x264:[0.99870,0.99754,0.99538,0.99116,0.98339,0.96956,0.94591,0.90982]
    # SSIM_x265:[0.99834,0.99681,0.99381,0.98815,0.97793,0.95972,0.93083,0.88696]
    # SSIM_vp9:[0.99923,0.99884,0.99815,0.99661,0.99335,0.98779,0.97765,0.95857]
    # MS_SSIM_x264:[0.99764,0.99578,0.99258,0.98665,0.97684,0.96112,0.93635,0.90133]
    # MS_SSIM_x265:[0.99703,0.99469,0.99047,0.98325,0.97107,0.95147,0.92243,0.88091]
    # MS_SSIM_vp9:[0.99856,0.99790,0.99677,0.99443,0.98986,0.98264,0.97059,0.94978]

####################################################
    #WITCHER3_data:
    #[15,19,23,27,31,35,39,43]
    #[14,21,28,35,42,49,56,63]
    #bitrate_x264:[80196180,44255355,23004356,11390812,5694853,2959360,1680399,1039163]
    #bitrate_x265:[38669289,20459025,10108623,4814208,2386496,1290028,739860,413353]
    #bitrate_vp9:[102899579,63998044,35857326,16426327,6460483,2722993,1201311,486028]
    # VMAF_x264:[98.14288,95.29824,90.17855,81.53154,70.28732,59.45350,49.96096,39.65963]
    # VMAF_x265:[93.75871,88.50309,81.17229,72.19445,62.20094,51.96991,41.75671,30.80241]
    # VMAF_vp9:[99.18291,98.05326,95.82821,91.36837,83.92734,75.37325,65.53597,54.33064]
    # PSNR_x264:[41.66310,39.10924,36.99854,35.23961,33.76506,32.57404,31.51497,30.36183]
    # PSNR_x265:[39.22061,37.11302,35.38523,34.03408,32.97152,32.01632,31.03920,29.83287]
    # PSNR_vp9:[43.66892,41.41469,39.29646,37.21812,35.46069,34.25141,33.26312,32.16062]
    # SSIM_x264:[0.99511,0.98967,0.97933,0.96008,0.92915,0.89678,0.87272,0.85331]
    # SSIM_x265:[0.98896,0.97901,0.96231,0.93844,0.91216,0.88839,0.86801,0.84632]
    # SSIM_vp9:[0.99703,0.99453,0.99006,0.98098,0.96512,0.94567,0.92266,0.89376]
    # MS_SSIM_x264:[0.99105,0.98280,0.96924,0.94813,0.91883,0.89020,0.86883,0.85149]
    # MS_SSIM_x265:[0.98185,0.96871,0.94987,0.92654,0.90315,0.88241,0.86423,0.84402]
    # MS_SSIM_vp9:[0.99442,0.99029,0.98352,0.97131,0.95278,0.93301,0.91206,0.88731]
if __name__ == '__main__':

    rate1 = np.array([38669289,20459025,10108623,4814208])
    psnr1 = np.array([93.75871,88.50309,81.17229,72.19445])
    
    rate2 = np.array([102899579,63998044,35857326,16426327])
    psnr2 = np.array([99.18291,98.05326,95.82821,91.36837])
    
    bd_psnr = BD_PSNR(rate1, psnr1, rate2, psnr2, piecewise=True)
    bd_rate = BD_Rate(rate1, psnr1, rate2, psnr2, piecewise=True)
    print(bd_psnr, bd_rate)
    bd_psnr1 = BD_PSNR(rate1, psnr1, rate2, psnr2, piecewise=False)
    bd_rate1 = BD_Rate(rate1, psnr1, rate2, psnr2, piecewise=False)
    print(bd_psnr1, bd_rate1)
    



    
    
    
