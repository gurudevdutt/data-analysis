import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

dir_path='/Users/gurudev/Dropbox/Pittsburgh/Lab/Analyzed Data/forErin/'

def getdata(filename):
    """ this function reads data from pulsed ESR experiment on Erin/Kai's setup.
    It takes a filename and returns (xaxis, percent signal, error) """
    try:
        my_file_handle = open(filename)
    except IOError:
        print("File not found or path is incorrect")
        raise
    finally:
        print("exit")
    data = np.genfromtxt(filename, delimiter='\t')
    # get the signal and reference
    tempsig = data[:,0]
    tempref = data[:,1]
    #process the metadata file
    metadatfile = filename.split(".txt")[0]+".log"
    mfilehandle=open(metadatfile)
    mdata = mfilehandle.readlines()
    mfilehandle.close()
    numavgs=int(mdata[3].split('\r\n')[0])
    timestart=float(mdata[1].split('\r\n')[0].split(',')[0].split('[')[1])
    timestop=float(mdata[1].split('\r\n')[0].split(',')[2].split(']')[0])
    timestep=float(mdata[1].split('\r\n')[0].split(',')[1])
    numpoints=mdata[4].split('\t').__len__()-1

    # the signal and refernce arrays need to be partitioned into lists
    # that are have numavg rows and numpoints columns
    tempsig = np.array_split(tempsig, numavgs)
    tempref = np.array_split(tempref, numavgs)
    #average along the rows
    signal = np.average(tempsig,axis=0)
    reference = np.average(tempref,axis=0)
    meanref = np.mean(reference)
    #calculate errors in signal and reference and then the x and y data
    sigerr=np.sqrt(signal)
    referr = np.sqrt(reference)
    xdata = np.linspace(timestart,timestep*len(signal),len(signal))
    ydata = (signal - reference)/meanref
    #error bar calculation is not perfect yet, error in meanref needs to
    #be taken into account
    errb = np.sqrt(sigerr**2 + referr**2)/meanref/np.sqrt(numavgs)
    # plt.errorbar(xdata,ydata,yerr=errb,fmt='o')
    # plt.show()
    return xdata,ydata,errb

def rabifunc(x, a1,a2,a3,a4,a5):
    return a1 + a2 * np.cos(2*np.pi*x/a3 + a4) * np.exp(-a5*x)

def fitrabi(x1,y1,yerr,pguess=[-0.1,0.1,25,0.0,0.1]):
    """ this function takes in data and fits it to a rabi function and then returns"""

    popt, pcov = curve_fit(rabifunc, x1, y1, sigma=yerr, p0=pguess,maxfev=10000)
    print("rabi fit parameters: ", popt)
    print("rabi fit uncertainties: ", np.sqrt(np.diag(pcov)))

    return popt,np.sqrt(np.diag(pcov))

def my_fitfunc(x, a, b):
    return a * x + b

def renorm(rabifile,datfile):
    rabixdat,rabiydat,rabiyerr  = getdata(rabifile)
    xdata, ydata, yerr = getdata(datfile)
    rabiguess = [-0.1,0.1,25,0.0,0.1]
    popt, perr = fitrabi(rabixdat,rabiydat,rabiyerr,rabiguess)

    bkgd = popt[0]
    contrast = popt[1]
    renormdat = 0.5+(ydata - bkgd)/(2*contrast)
    renormerr = yerr/(2*contrast)
    return xdata,renormdat,renormerr,popt,perr


if __name__ == "__main__":
    os.chdir(dir_path)
    rabifile = "square/Rabi_square_tsweep_0_100_1ns_300mv.txt"
    datfile = "gaussian/Gauss_2pi_resonance_2_845GHz.txt"

    x1, y1, yerrb, popt, perr= renorm(rabifile,datfile)

    x_plot = np.linspace(x1[0], x1[-1], 100)
    y_plot = 0.5 + (rabifunc(x_plot, *popt) - popt[0])/(2*popt[1])
    mpl.rcParams['text.usetex']=True

    f=plt.figure()
    #plt.plot(x_plot, y_plot, 'r')
    plt.errorbar(x1, y1-0.1, yerr=yerrb,fmt='o')
    #plt.plot(x_plot,y_plot,'r-')
    plt.xlabel(r'Number of pulses ($n$)',fontsize=16)
    plt.ylabel(r'Fidelity',fontsize=16)
    plt.xlim(0,19)
    plt.ylim(0,1.2)

    f.savefig('gaussian_fidelity_detuning_onres.pdf',bbox_inches='tight')

    # chi2 = np.sum(((y1 - y_plot) / yerrb) ** 2)
    # print("reduced chi2: ", chi2 / (y1.__len__() - len(popt)))
    # plot the residuals to see if there's any other frequency
    # plt.figure(2)
    # rabiresid = y1 - y_plot
    # plt.errorbar(x1, rabiresid, yerr=yerrb, fmt='gs')
    plt.show()




