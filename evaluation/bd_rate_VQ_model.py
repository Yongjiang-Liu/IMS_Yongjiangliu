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
    #[15,19,23,27,31,35,39,43]
    #[14,21,28,35,42,49,56,63]
    #bitrate_x264:[41997227,25807528,15751485,9619560,5937061,3778836,2441714,1599225]
    #bitrate_x265:[32418146,20388552,12456109,7510694,4449460,2622452,1500546,829564]
    #bitrate_vp9:[41806972,26822049,16164281,9147516,4837119,2798714,1679976,929824]
    # VMAF_x264:[97.94886,96.80327,94.24999,88.57425,79.26087,66.28396,49.90482,32.39038]
    # VMAF_x265:[97.90179,96.64668,93.92503,88.89962,80.81031,69.94987,56.64071,40.45569]
    # VMAF_vp9:[98.27806,97.90304,97.09047,95.14404,90.88382,84.81339,76.29832,63.51391]
    # PSNR_x264:[46.68796,44.21453,41.69414,39.03551,36.43507,33.87121,31.39086,29.30073]
    # PSNR_x265:[47.21540,45.06672,42.74506,40.31000,37.78229,35.27033,32.79255,30.32934]
    # PSNR_vp9:[48.47646,46.79088,44.98214,43.04359,40.72701,38.46000,36.12455,33.82528]
    # SSIM_x264:[0.99881,0.99768,0.99553,0.99091,0.98137,0.96185,0.92617,0.87953]
    # SSIM_x265:[0.99831,0.99674,0.99365,0.98770,0.97677,0.95773,0.92833,0.88719]
    # SSIM_vp9:[0.99887,0.99816,0.99690,0.99422,0.98899,0.98181,0.97035,0.95004]
    # MS_SSIM_x264:[0.99761,0.99564,0.99217,0.98584,0.97478,0.95501,0.92175,0.87991]
    # MS_SSIM_x265:[0.99692,0.99448,0.99020,0.98298,0.97121,0.95261,0.92557,0.88857]
    # MS_SSIM_vp9:[0.99798,0.99679,0.99481,0.99110,0.98450,0.97614,0.96407,0.94453]
    ########################################################################
    #DOTA2_data:
    #[15,19,23,27,31,35,39,43]
    #[14,21,28,35,42,49,56,63]
    #bitrate_x264:[58819486,36274278,21897270,13270741,8116840,5015056,3125819,2005908]
    #bitrate_x265:[43736237,27409549,16832580,10337876,6193960,3755455,2202473,1167293]
    #bitrate_vp9:[38661520,21964656,12097040,6411610,3236528,1774057,1019076,571469]
    # VMAF_x264:[97.57829,95.30093,91.43130,85.42111,75.72347,61.08757,43.48619,26.15053]
    # VMAF_x265:[97.05892,94.91604,91.55728,86.11557,77.53981,65.09972,50.00247,34.34606]
    # VMAF_vp9:[98.70748,97.76147,96.04681,92.92850,87.97404,81.46269,72.61664,59.88852]
    # PSNR_x264:[43.50666,41.02699,38.70444,36.42870,34.19697,31.94646,29.78636,27.94385]
    # PSNR_x265:[43.32597,41.16669,39.02057,36.89879,34.75218,32.62240,30.58172,28.61049]
    # PSNR_vp9:[45.04376,43.17771,41.36161,39.38412,37.40244,35.60903,33.73460,31.69585]
    # SSIM_x264:[0.99815,0.99653,0.99357,0.98747,0.97498,0.95063,0.91016,0.86403]
    # SSIM_x265:[0.99779,0.99613,0.99271,0.98597,0.97326,0.95101,0.91834,0.87469]
    # SSIM_vp9:[0.99810,0.99685,0.99499,0.99099,0.98417,0.97497,0.96062,0.93754]
    # MS_SSIM_x264:[0.99635,0.99360,0.98896,0.98064,0.96653,0.94274,0.90594,0.86456]
    # MS_SSIM_x265:[0.99582,0.99300,0.98783,0.97918,0.96515,0.94350,0.91390,0.87513]
    # MS_SSIM_vp9:[0.99661,0.99470,0.99203,0.98689,0.97874,0.96838,0.95357,0.93142]

#############################################
    #RUST_data:
    #[15,19,23,27,31,35,39,43]
    #[14,21,28,35,42,49,56,63]
    #bitrate_x264:[84960521,49839021,26890323,12952713,5816382,2820969,1588549,1060940]
    #bitrate_x265:[58011140,34036017,18319762,8704681,3891128,1821532,877271,442985]
    #bitrate_vp9:[101840723,66262027,38565706,19288013,7956792,3787349,1669348,667092]
    # VMAF_x264:[97.38041,94.44595,89.36847,80.70583,67.08964,51.22878,34.97454,20.10045]
    # VMAF_x265:[96.23105,92.67706,86.68013,77.35854,65.05730,51.42228,37.41998,21.82827]
    # VMAF_vp9:[98.77894,97.69839,95.37696,90.94877,82.81629,74.47084,64.06110,51.09725]
    # PSNR_x264:[42.25771,39.28655,36.70816,34.46299,32.62401,31.28715,30.27807,29.51502]
    # PSNR_x265:[40.91869,38.20397,35.79348,33.83401,32.35464,31.22521,30.34089,29.58209]
    # PSNR_vp9:[44.19752,41.72644,39.16219,36.67627,34.51708,33.24997,32.18728,31.16213]
    # SSIM_x264:[0.99730,0.99448,0.98926,0.97793,0.95410,0.92336,0.90050,0.88164]
    # SSIM_x265:[0.99614,0.99242,0.98494,0.97116,0.94957,0.92534,0.90443,0.88345]
    # SSIM_vp9:[0.99798,0.99655,0.99372,0.98722,0.97374,0.96001,0.94332,0.92136]
    # MS_SSIM_x264:[0.99489,0.98999,0.98131,0.96512,0.93848,0.90957,0.88891,0.87248]
    # MS_SSIM_x265:[0.99267,0.98619,0.97454,0.95671,0.93378,0.91113,0.89244,0.87400]
    # MS_SSIM_vp9:[0.99632,0.99380,0.98907,0.97910,0.96095,0.94473,0.92774,0.90766]
################################################################
    #STARCRAFT_data:
    #[15,19,23,27,31,35,39,43]
    #[14,21,28,35,42,49,56,63]
    #bitrate_x264:[47387588,32207480,21286117,13730659,8588427,5286585,3276382,2075134]
    #bitrate_x265:[37936199,25965088,17404007,11473753,7246813,4452681,2583181,1357757]
    #bitrate_vp9:[23139389,15691155,9942038,5736032,3016611,1711621,1000076,547829]
    # VMAF_x264:[97.49321,96.36437,93.82493,88.80831,80.81127,68.34149,51.12231,32.43100]
    # VMAF_x265:[97.52647,96.47665,94.28032,90.25101,83.53302,73.19870,58.49433,41.20763]
    # VMAF_vp9:[97.98038,97.68803,97.10925,95.59946,92.47204,87.91924,80.32051,68.87491]
    # PSNR_x264:[44.80435,41.88777,38.99979,36.18740,33.53664,31.07494,28.70837,26.61669]
    # PSNR_x265:[45.02145,42.37769,39.75031,37.19373,34.65103,32.17034,29.77265,27.52593]
    # PSNR_vp9:[47.67389,45.71348,43.41564,40.86190,38.25047,35.89133,33.50421,31.08416]
    # SSIM_x264:[0.99901,0.99806,0.99618,0.99227,0.98445,0.96856,0.93839,0.89244]
    # SSIM_x265:[0.99888,0.99788,0.99588,0.99195,0.98436,0.97043,0.94486,0.90254]
    # SSIM_vp9:[0.99920,0.99879,0.99812,0.99665,0.99370,0.98886,0.97956,0.96145]
    # MS_SSIM_x264:[0.99804,0.99638,0.99330,0.98737,0.97664,0.95777,0.92600,0.88137]
    # MS_SSIM_x265:[0.99785,0.99621,0.99318,0.98779,0.97824,0.96215,0.93524,0.89434]
    # MS_SSIM_vp9:[0.99855,0.99787,0.99675,0.99447,0.99021,0.98364,0.97191,0.95132]

####################################################
    #WITCHER3_data:
    #[15,19,23,27,31,35,39,43]
    #[14,21,28,35,42,49,56,63]
    #bitrate_x264:[92699207,50425666,25463667,12185986,5873816,2921328,1597681,975924]
    #bitrate_x265:[57652146,31363422,15796185,7533952,3525532,1742826,915900,486558]
    #bitrate_vp9:[120197876,73031318,40117563,18001918,6828357,2935948,1313726,564381]
    # VMAF_x264:[97.54295,94.25761,88.79783,80.38406,67.69064,54.10018,41.10107,28.63703]
    # VMAF_x265:[95.97277,92.18844,86.43336,78.52239,68.34805,56.86009,45.17680,33.07635]
    # VMAF_vp9:[99.07652,97.71132,94.89247,88.89795,80.70075,73.22172,64.49111,53.90571]
    # PSNR_x264:[42.08914,39.27128,36.98795,35.11681,33.50676,32.11290,30.88418,29.78392]
    # PSNR_x265:[40.73064,38.38057,36.36817,34.75440,33.46003,32.35768,31.34528,30.17604]
    # PSNR_vp9:[43.97324,41.45176,39.16558,36.88503,35.13754,34.07137,33.14458,32.06360]
    # SSIM_x264:[0.99640,0.99203,0.98363,0.96727,0.93607,0.89799,0.87044,0.84970]
    # SSIM_x265:[0.99338,0.98706,0.97579,0.95744,0.93077,0.90091,0.87728,0.85547]
    # SSIM_vp9:[0.99706,0.99421,0.98894,0.97676,0.95926,0.94296,0.92267,0.89550]
    # MS_SSIM_x264:[0.99311,0.98588,0.97391,0.95437,0.92396,0.89056,0.86638,0.84742]
    # MS_SSIM_x265:[0.98850,0.97932,0.96492,0.94469,0.91927,0.89319,0.87247,0.85258]
    # MS_SSIM_vp9:[0.99467,0.99013,0.98254,0.96730,0.94750,0.93057,0.91176,0.88825]
if __name__ == '__main__':
    rate1 = np.array([57652146,31363422,15796185,7533952])
    psnr1 = np.array([95.97277,92.18844,86.43336,78.52239])
    
    rate2 = np.array([120197876,73031318,40117563,18001918])
    psnr2 = np.array([99.07652,97.71132,94.89247,88.89795])
    
    bd_psnr = BD_PSNR(rate1, psnr1, rate2, psnr2, piecewise=True)
    bd_rate = BD_Rate(rate1, psnr1, rate2, psnr2, piecewise=True)
    print(bd_psnr, bd_rate)
    bd_psnr1 = BD_PSNR(rate1, psnr1, rate2, psnr2, piecewise=False)
    bd_rate1 = BD_Rate(rate1, psnr1, rate2, psnr2, piecewise=False)
    print(bd_psnr1, bd_rate1)
    



    
    
    
