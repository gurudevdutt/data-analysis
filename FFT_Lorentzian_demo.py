import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.fftpack as fftpack
import scipy.signal as signal

def lorentzian(x,bkg,amp,f0,w):
    return bkg + amp* np.power(w,2)/(np.power((x-f0),2)+np.power(w,2))




# this part demonstrates FFT of a square wave
# t = np.linspace(0,1,500,endpoint=False)
# y = signal.square(2*np.pi*5*t)
# sig_fft = fftpack.fft(y)
# N = len(t)
# f = fftpack.fftfreq(N, t[1] - t[0])
# power = np.abs(sig_fft)
# pos_mask = np.where(f > 0)
# freqs = f[pos_mask]
# peak_freq = freqs[power[pos_mask]>0]
# print('The peak frequencies are:', peak_freq)
# psd = power[pos_mask]
# mpl.rcParams['text.usetex']= True
# fig, ax = plt.subplots(2,tight_layout='True')
# ax[0].plot(t, y, 'b-',label=r'f(t)')
# ax[1].plot(freqs,psd,'r.-',label=r'$|F(\omega)|^2$')
# ax[0].legend(loc=0)
# ax[1].legend(loc=0)
# ax[0].set_xlabel(r'Time delay(s)', fontsize=16)
# ax[0].set_ylabel(r'Counts', fontsize=16)
# plt.show()

# this part demonstrates lorentzian fitting to a resonance
# to make it a bit chalenging I use data from a damped harmonic oscillator with noise
t = np.linspace(0,0.3,600,endpoint=False)
y = np.cos(2 * np.pi * 20 * t)* np.exp(- 4.0 *t)
N = len(t)
f = fftpack.fftfreq(N, t[1] - t[0])
# add in some noise
noise = np.random.normal(0,0.05,N)
y = y + noise

# find the FFT and choose only positive frequencies
sig_fft = fftpack.fft(y)
power = np.abs(sig_fft)
pos_mask = np.where(f > 0)
freqs = f[pos_mask]
# peak_freq = freqs[power[pos_mask]>0]
# print('The peak frequencies are:', peak_freq)
psd = power[pos_mask]
choosef = freqs[freqs<50.0]
choosep = psd[freqs < 50.0]
# fit the FFT to a lorentzian
popt,pcov = curve_fit(lorentzian,choosef,choosep,p0=[1,1,15.0,0.2],maxfev=10000)
print("Fit parameters:", popt)
print("Fit uncertainties:", np.sqrt(np.diag(pcov)))
x_fit = np.linspace(choosef[0],choosef[-1],300)
y_fit = lorentzian(x_fit,*popt)

mpl.rcParams['text.usetex']= True
fig, ax = plt.subplots(2,tight_layout='True')
ax[0].plot(t, y, 'b-',label=r'f(t)')
ax[1].plot(freqs[freqs<50.0],psd[freqs<50.0],'r.-',label=r'$|F(\omega)|^2$')
ax[1].plot(x_fit,y_fit,'',label = r'Fit')
ax[0].legend(loc=0)
ax[1].legend(loc=0)
ax[0].set_xlabel(r'Time delay(s)', fontsize=12)
ax[0].set_ylabel(r'Volts(V)', fontsize=12)
ax[1].set_ylabel(r'Power(W)', fontsize=12)
ax[1].set_xlabel(r'Hz', fontsize=12)
plt.show()




