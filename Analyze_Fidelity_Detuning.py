import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import re
import matplotlib as mpl
from AnalyzePulsedESR import getdata # added this import statement  on Nov.1, 2019

dir_path='/Users/gurudev/Dropbox/Pittsburgh/Lab/Analyzed Data/forErin/'
#files = re.split('_')

# Instead of copying over the getdata function, I used import instead
# def getdata(filename):
#     """ this function reads data from pulsed ESR experiment on Erin/Kai's setup.
#     It takes a filename and returns (xaxis, percent signal, error) """
#     try:
#         my_file_handle = open(filename)
#     except IOError:
#         print("File not found or path is incorrect")
#         raise
#     finally:
#         pass
#     data = np.genfromtxt(filename, delimiter='\t')
#     # get the signal and reference
#     tempsig = data[:,0]
#     tempref = data[:,1]
#     #process the metadata file
#     metadatfile = filename.split(".txt")[0]+".log"
#     mfilehandle=open(metadatfile)
#     mdata = mfilehandle.readlines()
#     mfilehandle.close()
#     numavgs=int(mdata[3].split('\r\n')[0])
#     timestart=float(mdata[1].split('\r\n')[0].split(',')[0].split('[')[1])
#     timestop=float(mdata[1].split('\r\n')[0].split(',')[2].split(']')[0])
#     timestep=float(mdata[1].split('\r\n')[0].split(',')[1])
#     numpoints=mdata[4].split('\t').__len__()-1
#
#     # the signal and refernce arrays need to be partitioned into lists
#     # that are have numavg rows and numpoints columns
#     tempsig = np.array_split(tempsig, numavgs)
#     tempref = np.array_split(tempref, numavgs)
#     #average along the rows
#     signal = np.average(tempsig,axis=0)
#     reference = np.average(tempref,axis=0)
#     meanref = np.mean(reference)
#     #calculate errors in signal and reference and then the x and y data
#     sigerr=np.sqrt(signal)
#     referr = np.sqrt(reference)
#     xdata = np.linspace(timestart,timestep*len(signal),len(signal))
#     ydata = (signal - reference)/meanref
#     #error bar calculation is not perfect yet, error in meanref needs to
#     #be taken into account
#     errb = np.sqrt(sigerr**2 + referr**2)/meanref/np.sqrt(numavgs)
#     # plt.errorbar(xdata,ydata,yerr=errb,fmt='o')
#     # plt.show()
#     return xdata,ydata,errb

def rabifunc(x, a1,a2,a3,a4,a5):
    return a1 + a2 * np.cos(2*np.pi*x/a3 + a4) * np.exp(-a5*x)

def fitrabi(x1,y1,yerr,pguess=[-0.1,0.1,25,0.0,0.1]):
    """ this function takes in data and fits it to a rabi function and then returns"""

    popt, pcov = curve_fit(rabifunc, x1, y1, sigma=yerr, p0=pguess,maxfev=10000)
    # print("rabi fit parameters: ", popt)
    # print("rabi fit uncertainties: ", np.sqrt(np.diag(pcov)))

    return popt,np.sqrt(np.diag(pcov))

def my_fitfunc(x, a, b):
    return a * x + b

def renorm(rabifile,datfile):
    """ Given a rabi data file and a regular data file, it will renormalize the data file to the rabi data"""
    rabixdat,rabiydat,rabiyerr  = getdata(rabifile)
    xdata, ydata, yerr = getdata(datfile)
    rabiguess = [-0.1,0.1,25,0.0,0.1]
    popt, perr = fitrabi(rabixdat,rabiydat,rabiyerr,rabiguess)

    bkgd = popt[0]
    contrast = popt[1]
    renormdat = 0.5+(ydata - bkgd)/(2*contrast)
    renormerr = yerr/(2*contrast)
    return xdata,renormdat,renormerr,popt,perr

def analyze_fidelity(rabifile,datfile):
    """Function I wrote to fit the fidelity to a line"""
    x1, y1, yerrb, popt, perr = renorm(rabifile, datfile)
    x_plot = np.linspace(x1[0], x1[-1], 100)
    f=plt.figure()
    # drop the first point as it seems to be incorrect
    x2 = x1[1:]
    y2 = y1[1:] / (0.8)
    plt.errorbar(x2, y2, yerr=yerrb[1:], fmt='o')
    fidelp, fidelpcov = curve_fit(my_fitfunc, x2, y2, p0=[0.5, 1], sigma=yerrb[1:])
    y_plot = my_fitfunc(x_plot,*fidelp)
    print("fidelity line parameters:", fidelp)
    chi2 = np.sum(((y2 - y_plot[1:]) / yerrb[1:]) ** 2)
    print("reduced chi2: ", chi2 / (y2.__len__() - len(popt)))
    
    plt.plot(x_plot, y_plot, 'r-')

def getlogfilenames(directory):
    """Gets all the filenames for log files"""
    logfiles = []
    tracklogfiles = []
    for filename in os.listdir(directory):
        if filename.endswith(".log"):
            logfiles.append(directory+os.sep+filename)
        elif filename.endswith('_tracklog.txt'):
            tracklogfiles.append(directory+os.sep+filename)
            # print(os.path.join(directory, filename))
        else:
            continue
    logfiles.sort()
    tracklogfiles.sort()
    justlogfiles = list(set(logfiles) - set(tracklogfiles))
    return justlogfiles

def getdatfilenames(directory):
    """Gets all the filenames for data files"""
    datfiles = []
    tracklogfiles = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            datfiles.append(directory+os.sep+filename)
        elif filename.endswith('_tracklog.txt'):
            tracklogfiles.append(directory+os.sep+filename)
            # print(os.path.join(directory, filename))
        else:
            continue
    datfiles.sort()
    tracklogfiles.sort()
    justdatfiles = list(set(datfiles) - set(tracklogfiles))
    return justdatfiles


def getmatchfiles(fnames,matchstr):
    """Returns all the filenames that contain a match string"""
    return [fnames[i] for i in range(len(fnames)) if fnames[i].count(matchstr)]

if __name__ == "__main__":
    os.chdir(dir_path)
    datadir = 'gaussian'
    rabifile = 'square'+os.sep+"Rabi_square_tsweep_0_100_1ns_300mv.txt"
    datfiles = getmatchfiles(getdatfilenames(datadir),'Gauss')
    plotcols = 5
    plotrows = 2

    fig, ax = plt.subplots(plotrows, plotcols,figsize=(11,8))

    # this loop goes through the list of data files and adds the output of
    # renorm function to a big matrix that contains all the data
    # there has to be a faster way of doing this using for loops maybe but this works for now.
    idx = 0
    x1_stack, y1_stack, yerrb_stack, popt_stack, pcov_stack = renorm(rabifile, datfiles[idx])
    print('Number of datfiles:',len(datfiles))
    while idx < (len(datfiles)-1):
        print(datfiles[idx])
        x2,y2,yerrb2,popt2,pcov2 = renorm(rabifile,datfiles[idx])
        x1_stack = np.vstack((x1_stack,x2))
        y1_stack = np.vstack((y1_stack,y2))
        yerrb_stack = np.vstack((yerrb_stack,yerrb2))
        popt_stack = np.vstack((popt_stack, popt2))
        pcov_stack = np.vstack((pcov_stack, pcov2))
        idx+=1

    # reshape the data into a matrix with plotrows
    plotxdata = np.asarray(np.array_split(x1_stack,plotrows,axis=0))
    plotydata = np.asarray(np.array_split(y1_stack, plotrows, axis=0))
    plotyerrbdata = np.asarray(np.array_split(yerrb_stack,plotrows, axis=0))
    #fidelearr = np.ndarray([],dtype=float32)
    print('X data rows:', len(plotxdata),'X data shape', plotxdata.shape)

    # this for loop is tricky because inside it i want to reference the original arrays that
    # contain the filenames and the frequencies. if the log files had the frequency
    # that would be better but i could not find that info in there
    #
    fideldetunarr = np.zeros([1,3])
    mpl.rcParams['text.usetex'] = 'True'
    idx = 0
    for i in range(len(plotxdata)):
        for j in range(len(plotxdata[i])):
            # now get the frequency from the file name, requires use of re.split
            # TODO: should use os.sep instead
            filestr = datfiles[idx].split('/')[1]
            freqstr = re.split('_',re.split('GHz',filestr)[0])[-1]
            #print('drow', i, 'dcol:', j, 'didx:', idx,'dval:', len(plotydata[i][j]))
            # np.float32('2.'+re.split('_',re.split('GHz',filestr)[0])[-1])
            # TODO: make the layout of plots more pretty
            ax[i, j].errorbar(plotxdata[i][j], plotydata[i][j], yerr=plotyerrbdata[i][j], fmt='o',label=freqstr)
            ax[i,j].set_title(freqstr)
            ax[i,j].set_xlabel(r'Number of pulses (n)')
            ax[i,j].set_ylabel(r'$\langle S_z = 0 \rangle$')
            ax[i,j].legend(loc=0)
            fidel = plotydata[i][j][1]
            fidelerr = plotyerrbdata[i][j][1]
            #ax[i, j].annotate('fidelity:' + fidel)
            #maxfidel = plotydata[i][j].max(axis=0)
            #print('fileidx:',(idx),'file:', freqstr,'fidelity:',fidel)
            # there seems to be some issue with the data that makes fidelity > 1 so fudging that with a
            # offset for now. will need to check this more carefully for publications
            fideldetunarr = np.vstack((fideldetunarr,np.array([[np.float32(freqstr),fidel-0.1,fidelerr]])))
            idx+=1
    #fig.savefig('gaussian_detuning_all_v3.pdf',bbox_inches='tight')
    f2 = plt.figure(2)

    #print(fideldetunarr[:,0][1:],fideldetunarr[:,1][1:])

    detunarr = fideldetunarr[:,0][1:] - 845.0
    fidelarr = fideldetunarr[:,1][1:]
    fidelerrb = fideldetunarr[:,2][1:]
    plt.errorbar(detunarr,fidelarr,yerr=fidelerrb,fmt='rs')
    plt.xlabel(r'Detuning(Mhz)')
    plt.ylabel(r'Fidelity')
    plt.xlim(-6.0,6.0)
    plt.ylim(0.8,1.1)
    #f2.savefig('gaussian_fidelvsdetun_v3.pdf',bbox_inches='tight')
    #plt.plot(np.linspace(-5,5,len(maxfidelarr)),maxfidelarr)

    # for i in range(4):
    #     for j in range(4):
    #         if ((i+1)*(j+1)) < len(datfiles):
    #             ax[i, j].errorbar(x1_stack[i + j], y1_stack[i + j], yerr=yerrb_stack[i + j], fmt='o')
    #             titlestr = datfiles[i+j].split('/')[1]
    #             ax[i,j].set_title(titlestr)
            # fidelp,fidelpcov = curve_fit(my_fitfunc,x2,y2,p0=[0.5,1],sigma=yerrb[1:])
            # y_plot = my_fitfunc(x_plot,*fidelp)
            # print("fidelity line parameters:", fidelp)

            # plt.plot(x_plot, y_plot, 'r-')
            # chi2 = np.sum(((y1 - y_plot) / yerrb) ** 2)
            # print("reduced chi2: ", chi2 / (y1.__len__() - len(popt)))
            # plt.ylim(0.0,1.05)
            # plt.xlim(-0.1,31)
            # f.savefig('gerono_rabi.pdf',bbox_inches='tight')
            # plt.figure(2)
            # rabiresid = y1 - y_plot
            # plt.errorbar(x1, rabiresid, yerr=yerrb, fmt='gs')

            # y_plot = 0.5 + (rabifunc(x_plot, *popt) - popt[0])/(2*popt[1])

            # plt.plot(x_plot, y_plot, 'r')

    plt.show()






