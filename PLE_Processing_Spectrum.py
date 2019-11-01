import numpy as np
import matplotlib.pylab as plt
import re
from scipy.optimize import curve_fit

#Modified by Gurudev on 2019-09-15
#print np.arange(637.00,639.00,0.08)

path="/Users/gurudev/Dropbox/Pittsburgh/Lab/Data/Duttlab3/Data/PLE/2019-07-05/NV1/"


import glob, os
os.chdir(path)
filename="637.15_1.0_-29.0_29.0_full.txt"
a=re.split('_',filename)


flag='GHz' # GHz, nm, V
dwell_time=0.001 #sec

def lorentzian(x,bkg,amp,f0,w):
    return bkg + amp*w**2/((x-f0)**2+w**2)

if flag=='nm':
    data = np.loadtxt(filename)
    dim = np.shape(data)
    print(dim)
    center=float(a[0])
    x0 = np.linspace(float(a[2]), float(a[3]), 800)
    y0 = sum(data) / dim[0]
    print(center)
    x=299792458.0/(x0*1E9+299792458.0/(center*1E-9))
    y=y0
    plt.plot(x*1E9,y/dwell_time,'.-',label='PLE Cps')
    plt.xlabel('nm')
    plt.ylabel('cps')
    plt.legend(loc=0)
if flag=='V':
    data = np.loadtxt(filename)
    dim = np.shape(data)
    print(dim)
    x = np.linspace(float(a[2])/-10.0, float(a[3])/-10.0, 800)
    y = sum(data) / dim[0]
    plt.plot(x, y / dwell_time, '.-', label='PLE Cps')
    plt.xlabel('V')
    plt.ylabel('cps')
    plt.legend(loc=0)
if flag == 'GHz':
    data = np.loadtxt(filename)
    dim=np.shape(data)
    print(dim)
    x = np.linspace(float(a[2]), float(a[3]), 800)
    y=sum(data)/dim[0]/dwell_time
    errb = np.sqrt(y)
    lowerx = 5.0
    upperx = 15.0
    choosex = x[np.logical_and(x >= lowerx, x < upperx)]
    choosey = y[np.logical_and(x >= lowerx, x < upperx)]
    chooseyerr = errb[np.logical_and(x >= lowerx, x < upperx)]

    popt,pcov = curve_fit(lorentzian,choosex,choosey,p0=[100,5000,10.0,0.2],maxfev=10000)
    print("Fit parameters:", popt)
    print("Fit uncertainties:", np.sqrt(np.diag(pcov)))
    x_fit = np.linspace(choosex[0],choosex[-1],300)
    y_fit = lorentzian(x_fit,*popt)
    #
    f = plt.figure()
    plt.errorbar(choosex,choosey,yerr=chooseyerr,fmt='bo',label='T = 18 K')
    plt.plot(x_fit, y_fit, 'r-',label='Fit')
    #plt.plot(choosex,lorentzian(choosex,100,4000,10.0,0.2),'.-')
    plt.xlabel('Laser detuning (GHz)')
    plt.ylabel('Counts/sec')
    plt.legend(loc=0)
    #f.savefig("nv1_ple.pdf",bbox_inches='tight')
plt.show()

