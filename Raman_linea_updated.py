# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:22:43 2023

@author: msinloz. TODO BASADO EN EL TRABAJO DE A. Boscá

########
Created on 20/02/2019
Modified 05/07/2019
Script for batch-process raman spectra
Basic usage:
    --remove baseline
    --get an initial model
    --get files with spectra
    --apply model to spectra
    --save exit
1p1: Added a config file to move all the data and allow repeatability
1p2: Fixed output files
1p3: Strain and doping maps (VectoresLee2012). Modified I values
 Imax = A/(pi*fwhm), removed multiplied by 2 fwhm
1p4: Possibility to remove cosmic Rays.
Changed graphene mask to use amplitudes
Changed no convergence values to 1 (avoid division by zero)
@author: albosca
#######

20220512: Cambio de nombre del archivo: Raman_lineas
20220513: Introduzco modificación de los standares de pintar modificando .rc

20230323: Updates the way of working to have everything in a SINGLE PYTHON Project
20230323: Changes the process of getting the background removal: 
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    
    [1] S.-J. Baek, A. Park, Y.-J. Ahn, y J. Choo, «Baseline correction using asymmetrically reweighted penalized least squares smoothing», Analyst, vol. 140, n.º 1, pp. 250-257, 2015, doi: 10.1039/C4AN01061B.

"""

import numpy as np
from numpy.linalg import norm

import scipy.constants as phys_constants
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.pyplot import (plot,xlabel,ylabel,savefig,clf,title,figure,
                                imshow,contour,colorbar) 
import scipy.optimize as optimize
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve

import os, time
import re
import tkinter
from tkinter import filedialog

from numpy.linalg import norm

from Define_parameters import pos_boundaries




################################################

plt.rc('lines', linewidth=1., markersize=5)
plt.rc('grid', linewidth=0.5, ls='--', c='k')
plt.rc('xtick', direction='in',top='True',labelsize=10)
plt.rc('ytick', direction='in',right='True',labelsize=10)
plt.rc('font',family='serif')
# plt.rc('legend', numpoints=1,)


##################################################

#######################

#### Define the folders: Where files to be analyzed are and the folder where the analyzed files will be

#######################

folder_to_be_analyzed = r'To_be_analyzed_lines'

folder_to_save = r'Analysis'

#######################


#### Defined parameters for the analysis


deffolder = "D:\\Datos"
deffilename = "testraman.txt"
#Depends on laser frequency
#Origin values
defPosG0 = 1581.6 # in cm-1. For lambda = 514.5nm 1581.6 cm-1
defPos2D0 = 2669.9  # in cm-1. For lambda = 514.5nm 2676.9 cm-1
mafra2007_wG = -18 #cm-1/eV excitation
mafra2007_w2D = +88 #cm-1/eV excitation
#lambda = 473nm 2695.5 cm-1 
#lambda = 532nm 2669.9 cm-1 
#Tensile strain values
defSlopeT = 2.2 #adim w2D/wG
#Hole doping values
defSlopeH = 0.70 #adim w2D/wG
#Conversion factors
defEpsilontocm1 = -23.5 #-69.1 biaxial -23.5 uniaxial# cm-1/%
defDopingtocm1 = -1.04/1e12 # cm-1/n(cm-2)/

defFolder = '/home/albosca/Documentos/peak-o-mat DATA/matrix/'
deffileToLoad = '/home/albosca/Documentos/peak-o-mat DATA/7 picos.lpj'
deffolderToSave = '/home/albosca/Documentos/peak-o-mat DATA/'
deffilenameToSave = 'load'
defmatrixToLoad = '/home/albosca/Documentos/peak-o-mat DATA/XY 50X  LS30 Pol 0 grados_003_(Sub BG) (Sub BG).txt'
defspectrumToLoad = '/home/albosca/Documentos/peak-o-mat DATA/RamanTest2.dat'
laserExcitation = 532.224 #nm
cmTonm = 1e7
posXfile = 0
posYfile = 0
debug = False  #verbose mode if true
saveToFile = True
numofiterations = 1000
xaxis_min = 1150 #cm-1
xaxis_max = 3500 #cm-1
numberofcols = 45
numberofrows = 45


initial_values, boundaries_tuple = pos_boundaries()

print (initial_values)
###############
#Función para comenzar a hacer el fitting
###############

def multiLorentzians(x, position, fwhm_simple, amp):
    """Fit to any number of lorenzians"""
    if (len(position)!=len(fwhm_simple) or len(position)!=len(amp) or len(amp)!=len(fwhm_simple)):
        print("Invalid number of parameters")
        return 0
    ypeaks = 0
    for i, pos in enumerate(position):
        ypeaks += 2*amp[i]*fwhm_simple[i]/(np.pi*(4*((x-pos)**2)+fwhm_simple[i]**2))  #Same as in origin and witec program
        # print(ypeaks)
    return ypeaks


def sevenLorentzians(x,p0,p1,p2,p3,p4,p5,p6,w0,w1,w2,w3,w4,w5,w6,a0,a1,a2,a3,a4,a5,a6):
    """Wrapper for multilorenzians"""
    return multiLorentzians(x,[p0,p1,p2,p3,p4,p5,p6],
                            [w0,w1,w2,w3,w4,w5,w6],
                            [a0,a1,a2,a3,a4,a5,a6])
###############
#To remove the background: Older function is backgroun Removal

###############
defRemoveCosmicRays = True
defRemoveBackground = True
defCheckRemoveBackground = True
xaxis_in_nm = False
defIntervalBG = [800., 1200., 2000., 2200., 3100., 3500.]
defIntervalSize = 30
deffilter_gradient = 100 # For removing points with large derivative (cosmic ray spikes)

# def backgroundRemoval(x,y, bgpoints = defIntervalBG, bgsize = defIntervalSize,
#                       plot_check = defCheckRemoveBackground, save_folder = deffolderToSave):
#     #Needs the x and y to remove. Improved for removing spikes

#     quadratic_func = lambda x,a,b,c: a+b*x+c*(x**2)
#     #Create a mask to select only certain regions for BG
#     mask = x>1e6 # Generate False array
#     for i in bgpoints:
#         mask = np.logical_or(mask,((i<x)*(x<i+bgsize))) # Select true only defined regions
#     #Fit the curve
#     fitted_val, fitted_cov = optimize.curve_fit(quadratic_func, x[mask], y[mask], p0 = [0.1,0.001,0.0001])
#     print (fitted_val, fitted_cov)
#     x_bg = x
#     y_bg = quadratic_func(x,*fitted_val)
#     if plot_check:
#         fig = figure(1)
#         fig.clf()
#         axes = fig.add_subplot(111)
#         axes.set_xlabel("cm-1")
#         axes.set_ylabel("Intensity (a.u)")
#         axes.set_title('Check Background removal')
#         axes.set_xlim(xaxis_min,xaxis_max)
#         #Autoscale the y axis in the range
#         y_limited = y[(xaxis_min<x)*(x<xaxis_max)]
#         axes.set_ylim(y_limited.min(),y_limited.max())
#         axes.plot(x, y, 'go')
#         axes.plot(x_bg, y_bg, 'r-')
#         axes.autoscale_view(tight=None, scalex=False, scaley=True)
#         savefig(save_folder+'_bg.svg',dpi = 500)
        
#     return x, y-y_bg

###############
###############

def baseline_arPLS(y, ratio=1e-6, lam=10, niter=10, full_output=False):
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values

        count += 1

        if count > niter:
            print('Maximum number of iterations exceeded')
            break

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        
    
        return z, d, info
    else:
        return z


def loadMatrix(matrixToLoad = defmatrixToLoad, xaxis_in_nm = False):
    """Load data directly from witec file (export as table, check units!)"""
    #Prepare the data
    spectraMatrix = np.loadtxt(matrixToLoad)
    #Convert to cm-1
    
    if xaxis_in_nm:
        xspectra = cmTonm/laserExcitation-cmTonm/spectraMatrix[:,0] 
    else:
        xspectra = spectraMatrix[:,0] 
    return xspectra,spectraMatrix


def lasershift(posG_init, pos2D_init, lambda_init, lambda_end):
    """Function to shift peak position in cm-1 as if it were measured with
    a different laser exitation (both in nm)
    References:
    Mafra, D. L.,  (2007). Determination of LA and TO phonon dispersion 
    relations of graphene near the Dirac point by double resonance 
    Raman scattering. Physical Review B, 76(23), 233407. 
    https://doi.org/10.1103/PhysRevB.76.233407

    Costa, S. D., (2012). Resonant Raman spectroscopy of graphene grown on 
    copper substrates. Solid State Communications, 152(15), 1317–1320. 
    https://doi.org/10.1016/j.ssc.2012.05.001

    Casiraghi, C.,... (2007). Raman fingerprint of charged impurities in 
    graphene. Applied Physics Letters, 91(23), 12–14. 
    https://doi.org/10.1063/1.2818692
    """
    inv_lambda_nm_to_ev = phys_constants.h*phys_constants.c/(1e-9*phys_constants.e)
    print(inv_lambda_nm_to_ev)
    posG_end = posG_init + mafra2007_wG*(1./lambda_end-1./lambda_init)*inv_lambda_nm_to_ev
    pos2D_end = pos2D_init + mafra2007_w2D*(1./lambda_end-1./lambda_init)*inv_lambda_nm_to_ev

    return posG_end, pos2D_end

#print(lasershift(1581.6,2669.9,532,473))
def generateUnitaryVectors(PosG0 = defPosG0, Pos2D0 = defPos2D0,
                           SlopeH = defSlopeH, SlopeT = defSlopeT):
    """For generating the vector base in our case. Returns eH and eT"""
    Vector0 = np.array([PosG0,Pos2D0])
    VectorH = np.array([PosG0+1,SlopeH*1+Pos2D0]) #to positive values
    VectorT = np.array([PosG0-1,SlopeT*(-1)+Pos2D0]) #to negative values
    #Create unitary vectors
    eH = (VectorH-Vector0)/np.linalg.norm((VectorH-Vector0))
    eT = (VectorT-Vector0)/np.linalg.norm((VectorT-Vector0))
    #Generate matrix for 
    changeBaseMatrix = np.array([eH,eT]).T#.I
#    print (Vector0, eT, eH, np.linalg.norm(eT), np.linalg.norm(eH))
#    print (eT,eH,changeBaseMatrix)
    return Vector0,eH,eT,changeBaseMatrix

defVector0,defeH,defeT,defchangeBaseMatrix= generateUnitaryVectors() #calculate standard vectors

def calculateDopingEpsilon(posGCalc = 1590, pos2DCalc = 2700, 
                           Vector0 = defVector0, 
                           Epsilontocm1 = defEpsilontocm1,
                           Dopingtocm1 = defDopingtocm1,
                           changebase = defchangeBaseMatrix):
    """For calculating the strain and doping values for a certain base"""
    VectorP = np.array([posGCalc,pos2DCalc])-Vector0
    unitary_eH_eT = np.linalg.inv(changebase).dot(VectorP)
    doping = changebase.dot([unitary_eH_eT[0],0])[1]/Dopingtocm1 #separates both components and then multiplies dw2D
    strain = changebase.dot([0,unitary_eH_eT[1]])[0]/Epsilontocm1#separates both components and then multiplies dwG
    print(posGCalc,pos2DCalc,doping,strain)
    return doping,strain

def calculatePosGPos2D(doping = 0.,strain= 0., 
                           Vector0 = defVector0, 
                           Epsilontocm1 = defEpsilontocm1,
                           Dopingtocm1 = defDopingtocm1,
                           changebase = defchangeBaseMatrix):
    """For calculating the G and 2D position values for strain and doping"""
    eH = changebase.dot([1,0])
    eT = changebase.dot([0,1])
    numberof_eH = Dopingtocm1 * doping / eH[1]
    numberof_eT = Epsilontocm1 * strain / eT[0]
    [posGCalc,pos2DCalc] = numberof_eH*eH + numberof_eT*eT + Vector0
    print(posGCalc,pos2DCalc,doping,strain)
    return posGCalc,pos2DCalc

def generateLines(posGrange = [1570,1610],
                         initposG = 1582, initpos2D = 2678,
                         Vector0 = defVector0, slope = defSlopeH,
                         folder = deffolder, filename = deffilename):
    """For plotting lines of constant Doping or strain"""
    xarray = np.linspace(posGrange[0],posGrange[1])
    yarray = initpos2D + (xarray-initposG)*slope
    np.savetxt(folder+filename, np.transpose([xarray,yarray]))
#    print (xarray,yarray)
    # plot(xarray,yarray,'r')

    
def cosmicRayRemoval(x,y, filter_gradient= deffilter_gradient):
    """Function to remove spikes. Needs to give mean value instead o zero"""
     #First spike-removal part. Check gradient
    y_derivative_info = np.abs(np.gradient(y))>filter_gradient #True/false array for region with spikes
    y_fixed = y
    #print(y_derivative_info)
    for distance in [3,2,1]: #Fix several times, in case there are two bad points next to each other
        for position,isSpike in enumerate(y_derivative_info):
            if (isSpike == True):
                try:
#                    print("Detected spike in position ",position)
                    y_fixed[position]= (y_fixed[position-distance]+y_fixed[position+distance])/2.0
                except:
#                    print("Failed spike removal in position ",position)
                    pass
    return x,y_fixed    


      
def getGrapheneMask(AG, FWG, num_of_sigmas = 3):
    """Return a true/false array with graphene/no graphene information"""
    print("Creating graphene mask")
    AGarray = np.array(np.nan_to_num(AG,False)) # Array to decide if there is graphene or not
    AGarray_gr = AGarray[(AGarray<(AGarray.mean()*10))*(AGarray>(AGarray.mean()*0.1))] # Subset with initial graphene
    accepted_interval = [AGarray_gr.mean()-AGarray_gr.std()*num_of_sigmas,AGarray_gr.mean()*3+AGarray_gr.std()*num_of_sigmas] #up to three layer
    isgraphenemask_AG=((AGarray>accepted_interval[0])*(AGarray<accepted_interval[1]))
    
    num_of_sigmas+=1 # less restrictive for second layer
    FWGarray = np.array(np.nan_to_num(FWG,False))
    FWGarray_gr = FWGarray[isgraphenemask_AG]
    accepted_interval = [FWGarray_gr.mean()-FWGarray_gr.std()*num_of_sigmas,FWGarray_gr.mean()+FWGarray_gr.std()*num_of_sigmas]
    isgraphenemask_FWG = ((FWGarray>accepted_interval[0])*(FWGarray<accepted_interval[1]))
    return isgraphenemask_FWG * isgraphenemask_AG

def getLayersMask(AG, FWG, num_of_sigmas = 3):
    """Return a true/false array with two layers graphene/no graphene information"""
    AGarray = np.array(AG)
    layers_mask = np.zeros_like(AG) #Initially all without graphene
    g_mask = getGrapheneMask(AG,FWG, num_of_sigmas) #get where graphene is present
    layers_mask += np.ones_like(AG)*g_mask # First layer of graphene
    bin_number = int(AG[g_mask].size/10.0)
    histogram_gr, bins_gr = np.histogram(AG[g_mask],bin_number)
    max_from_histogram = bins_gr[histogram_gr.argmax()] # get 1 layer center
    #Add second layer
    accepted_interval_2L = [max_from_histogram*2-AG[g_mask].std(),max_from_histogram*2+AG[g_mask].std()]
    layers_mask += np.ones_like(AG)*g_mask*((AGarray>accepted_interval_2L[0]))#*(IGarray<accepted_interval_2L[1]))
    #Add trhee layers and more
    accepted_interval_3plus = [max_from_histogram*3-AG[g_mask].std(),max_from_histogram*10]
    layers_mask += np.ones_like(AG)*g_mask*((AGarray>accepted_interval_3plus[0])) #*(IGarray<accepted_interval_3plus[1]))
    return layers_mask

def getCenter(AG, FWG, num_of_sigmas = 3):
    g_mask = getGrapheneMask(AG,FWG, num_of_sigmas) #get where graphene is present
    num_of_rows = AG[:][0].size
    num_of_cols = AG[0][:].size
    center_x = 0
    center_y = 0
    gr_points = 0
    for i in range(num_of_rows):
        for j in range(num_of_cols):
            if g_mask[i][j]:
                center_x += i
                center_y += j
                gr_points += 1.0
    return [center_x*1.0/gr_points, center_y*1.0/gr_points]

def saveMatrixPlot(fileToLoad =deffileToLoad, micron_per_pixel = 1):
    file_format = '.svg'
    matrixToPlot = np.loadtxt(fileToLoad)
    print ('mmmm',type(matrixToPlot),matrixToPlot.size)
    

    if(micron_per_pixel!=1):
        file_format = '_sc' + file_format
    x_axis = np.linspace(0,(matrixToPlot[0,:].size-1)*micron_per_pixel, matrixToPlot[0,:].size)
#    print("x axis:", x_axis)
    y_axis = np.linspace(0,(matrixToPlot[:,0].size-1)*micron_per_pixel, matrixToPlot[:,0].size)
#    print("y axis:", y_axis)
    figure(4)
    clf()
    #max and min values
    min_data = 0
    max_data = 0
    mean_data = 0
    std_data = 0
    sigma_filter = 6
    filtered_matrix = matrixToPlot[matrixToPlot!=0.0]
    try:
        min_data = filtered_matrix.min()
        max_data = filtered_matrix.max()
        mean_data = filtered_matrix.mean()
        std_data = filtered_matrix.std()
        if (abs(min_data) < abs(mean_data-sigma_filter*std_data) or 
            abs(max_data) > abs(mean_data+sigma_filter*std_data)):
            min_data = np.min([min_data,mean_data-sigma_filter*std_data])
            max_data = np.max([max_data,mean_data+sigma_filter*std_data])
    except:
        min_data = matrixToPlot.min()
        max_data = matrixToPlot.max()
    
    imshow(matrixToPlot,extent=[x_axis[0],x_axis[-1],y_axis[-1],y_axis[0]],
           interpolation ='bicubic',vmin=min_data,vmax=max_data)
    contour(x_axis,y_axis,matrixToPlot, origin='upper',linewidths=0.8, 
            vmin=min_data,vmax=max_data)
    imshow(matrixToPlot,extent=[x_axis[0],x_axis[-1],y_axis[-1],y_axis[0]], 
           interpolation ='bicubic',vmin=min_data,vmax=max_data)
    colorbar(shrink=0.8) #, extend='both'
    savefig(fileToLoad[:-4]+file_format,dpi = 500)

def saveallMatrixFiles(foldertosave,matrixlists, matrixnames):
    """Needs the folder where to save and the matrixes in order
    without the extension"""
    print("Saving .mtx files")
    for i,j in zip(matrixlists,matrixnames):
        np.savetxt(os.path.join(foldertosave,j+".mtx"),i.T) #modified, already an array
    
def saveallMatrixPlots(folderToLoad, micron_per_pixel = 1):
    """Draw matrices in SVG in the same folder"""
    if os.path.isdir(folderToLoad):
        filenames = os.listdir(folderToLoad)
    else:
        return
    for single_file in filenames:
        if single_file[-4:] == '.mtx':
            saveMatrixPlot(os.path.join(folderToLoad,single_file), micron_per_pixel)
    
def fitPeaks(x, y, filenameToSave = deffilenameToSave, peak_type = 'lorentz',
             peak_number = 7, initial_values = initial_values, 
             boundaries = boundaries_tuple, 
             remove_CosmicRays = defRemoveCosmicRays,
             remove_background = defRemoveBackground, 
             check_background = defCheckRemoveBackground ):
    """Function to calculate the broad peaks. Use x and y for the spectra.
    peak_type: 'lorenz', 'BWF'  """
    if remove_CosmicRays:
        x,y = cosmicRayRemoval(x,y)
    #---- Secondly remove background
    if remove_background:
        y_old = y
        
        z , y , info = baseline_arPLS(y, ratio=1e-6, lam=1e5, niter=100, full_output=True)
        
        fig, (ax) = plt.subplots(1, 1)
        fig.set_size_inches(6.692913385826771, 4.136447244094488)
        ax.plot(x,y_old)
        ax.plot(x,z, color= 'r')
        ax.plot(x,y, color= 'k')
        

        
        
        fig.savefig(filenameToSave+'_Background_removal.svg', format='svg')
        
        plt.close()
        
    #----Define plot to use--
    #----Define plot to use--
    
    fig, (ax1, ax2) = plt.subplots(1,2,sharey=False)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)
    
    ax1.plot(x, y, color='tab:purple', marker='P', markevery=60, zorder = 1)
    ax2.plot(x, y, color='tab:purple', marker='P', markevery=60, zorder = 1)
    
   
    ax1.set_xlim(1000, 1800)  # rango D y G
    ax2.set_xlim(2400, 3200) 
   
    y_limited = y[(xaxis_min<x)*(x<xaxis_max)]
    ax1.set_ylim(y_limited.min(),y_limited.max())
    ax2.set_ylim(y_limited.min(),y_limited.max())
    

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    ax1.xaxis.set_ticks_position('both')    
    ax2.xaxis.set_ticks_position('both')
    
    ax1.yaxis.set_ticks_position('left')
    ax2.yaxis.tick_right()
    
   
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((-d, +d), (-d, +d), **kwargs)  # bottom-right diagonal
    
    fig.subplots_adjust(wspace=0.04)
    


    #----Define fitting with the model
    if saveToFile:
        np.savetxt(filenameToSave + 'initial.dat', np.c_[x,y])
        savefig(filenameToSave + '_chk.svg',dpi = 400)
    try:
        xmodel = x # Keep x the same as the original data
        if (peak_type == 'lorentz'):
            fitted_values, fitted_cov = optimize.curve_fit(sevenLorentzians, x, y, bounds = boundaries, p0 = initial_values)
            ymodel = sevenLorentzians(xmodel, *fitted_values)
            fitted_error = np.sqrt(np.diag(fitted_cov))
            y_limited = np.append(ymodel[(xaxis_min<x)*(x<xaxis_max)],y[(xaxis_min<x)*(x<xaxis_max)]) # resize y axis
        # elif (peak_type == 'BWF'):
        #     fitted_values, fitted_cov = optimize.curve_fit(sevenBWF, x, y, bounds = boundaries, p0 = initial_values)
        #     ymodel = sevenBWF(xmodel, *fitted_values)
        #     fitted_error = np.sqrt(np.diag(fitted_cov))
        #     y_limited = np.append(ymodel[(xaxis_min<x)*(x<xaxis_max)],y[(xaxis_min<x)*(x<xaxis_max)]) # resize y axis
    
        else:
            print("Wrong peak type, check options")
            fitted_values = np.empty((peak_number*3))
            fitted_values.fill(np.nan)
            fitted_values = list(fitted_values)
            fitted_error= fitted_values
            xmodel = x
            ymodel = np.empty(len(x))
            ymodel.fill(np.nan)
            ymodel = list(ymodel)       
            y_limited = y[(xaxis_min<x)*(x<xaxis_max)] # resize y axis        
    except:
        #Modified to generate nans
        print("out of convergence, check raw data")
        print(y)
        fitted_values = np.empty((peak_number*3))
        fitted_values.fill(np.nan)
        fitted_values = list(fitted_values)
        fitted_error= fitted_values
        xmodel = x
        ymodel = np.empty(len(x))
        ymodel.fill(np.nan)
        ymodel = list(ymodel)       
        y_limited = y[(xaxis_min<x)*(x<xaxis_max)] # resize y axis
    
        

    ax1.plot(xmodel,ymodel,color = '#377eb8', zorder = 2, alpha = 0.7)
    ax2.plot(xmodel,ymodel,color = '#377eb8', zorder = 2, alpha = 0.7)
    
    ax1.set_ylim(y_limited.min(),y_limited.max())    #resize y axis
    ax2.set_ylim(y_limited.min(),y_limited.max())    #resize y axis

    for i in range(peak_number):
#        figure(2) # To check only the peaks
        if (peak_type == 'lorentz'):
            ymodelpeak = multiLorentzians(xmodel,[fitted_values[i]],
                                          [fitted_values[i+peak_number]],
                                          [fitted_values[i+2*peak_number]])
        if (peak_type == 'BWF'):
            ymodelpeak = multiBWF(xmodel,[fitted_values[i]],
                                          [fitted_values[i+peak_number]],
                                          [fitted_values[i+2*peak_number]],
                                          [fitted_values[i+3*peak_number]])
        ax1.plot(xmodel,ymodelpeak,color = '#ff7f00', zorder = 3, alpha = 0.5)
        ax2.plot(xmodel,ymodelpeak,color = '#ff7f00', zorder = 3, alpha = 0.5)
    
    #----Return parameters
#    tableResults = modelLoaded.parameters_as_table(witherrors=True)
    if saveToFile:
        #print(fitted_values)
        #print(fitted_cov)
        np.savetxt(filenameToSave + 'fit.dat', np.c_[xmodel,ymodel]) 
        np.savetxt(filenameToSave + '.csv', np.c_[fitted_values,fitted_error]) 
#        f = file(filenameToSave+'.csv','w')
#        f.write(modelLoaded.parameters_as_csv(witherrors = True))
#        f.close



        ax1.xaxis.set_major_locator(ticker.MultipleLocator(200))
    
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax2.axes.yaxis.set_ticklabels([])
    
    
        ax1.set_ylabel(r"Intensity$\thinspace$(arb.units)")
        fig.text(0.5, 0.01, 'cm$^{-1}$', ha='center', va='bottom', rotation='horizontal')

        ax2.legend(([r'Measurement',
                r'Fitting',
                r'Peak fitting'
                ]), loc='upper right')
        
        
        savefig(filenameToSave + '.svg',dpi = 400)
        plt.close()
#    if debug:
#        for row in tableResults:
#            print(row[:])
#        show() #plot the graph
        
    return fitted_values, fitted_error #tableResults,modelLoaded
    
def loadFromFit(folderToLoad, row_number = numberofrows, col_number = numberofcols, peak_type = 'lorentz'):
    """" Function to load the already fitted points inside a matrix folder.
    Extracts the graphs in .svg and .mtx files for all the points
    peak_type: 'lorentz', 'BWF' or 'BWF_2pk'"""
    filenames = os.listdir(folderToLoad)
    #Initialize matrix
    filenamematrix = []
    for i in range(col_number):
        row = []
        for j in range(row_number):
            row.append([])    
        filenamematrix.append(row)
            
    for singlefilename in filenames:
        if singlefilename[-4:] == '.csv':
            positions = re.findall(r'\d+',singlefilename)
            table = np.loadtxt(folderToLoad+singlefilename)#,dtype='string',delimiter =",")
            filenamematrix[int(positions[0])][int(positions[1])] = table #save output for each pixel
    
    print("Final Matrix")
#    print(filenamematrix)
    AD, AG, ADp, A2D = [], [], [], [] #For other calculations, such as "is graphene?" mtx
    ID, IG, IDp, I2D = [], [], [], []
    IDvsIG, IDpvsIG, I2DvsIG = [], [], []   
    P2D, PD, PDp, PG = [], [], [], []
    FW2D, FWD, FWDp, FWG = [], [], [], []
    Strain, Doping = [], []
    #numbers to easily change matrix positions
    peak_num = 7
    pos_num, w_num, A_num = [0*peak_num, 1*peak_num, 2*peak_num]
    D_num, G_num, Dp_num, TwoD_num = [0,1,2,4]
    for i in range(col_number):
        rowAD, rowAG, rowADp, rowA2D = [], [], [], []
        rowID, rowIG, rowIDp, rowI2D = [], [], [], []
        rowIDvsIG, rowIDpvsIG, rowI2DvsIG = [], [], []
        rowP2D, rowPD, rowPDp, rowPG = [], [], [], []
        rowFW2D, rowFWD, rowFWDp, rowFWG = [], [], [], []
        rowStrain, rowDoping = [], []
        for j in range(row_number):
            print("Row,Col:", j, i, "Row in file:", j*col_number+i+1)
            table = filenamematrix[i][j]
#            print(table)
            rowAD.append(float(table[A_num+D_num][0]))
            rowAG.append(float(table[A_num+G_num][0]))
            rowADp.append(float(table[A_num+Dp_num][0]))
            rowA2D.append(float(table[A_num+TwoD_num][0]))
            
            if(peak_type == 'lorentz'):
                rowID.append(float(table[A_num+D_num][0])*2/(np.pi*float(table[w_num+D_num][0])))
                rowIG.append(float(table[A_num+G_num][0])*2/(np.pi*float(table[w_num+G_num][0])))
                rowIDp.append(float(table[A_num+Dp_num][0])*2/(np.pi*float(table[w_num+Dp_num][0])))
                rowI2D.append(float(table[A_num+TwoD_num][0])*2/(np.pi*float(table[w_num+TwoD_num][0])))           
                rowIDvsIG.append((float(table[A_num+D_num][0])/float(table[A_num+D_num][0]))
                /(float(table[A_num+G_num][0])/float(table[w_num+G_num][0])))
                rowIDpvsIG.append((float(table[A_num+Dp_num][0])/float(table[A_num+Dp_num][0]))
                /(float(table[A_num+G_num][0])/float(table[w_num+G_num][0])))
                rowI2DvsIG.append((float(table[A_num+TwoD_num][0])/float(table[A_num+TwoD_num][0]))
                /(float(table[A_num+G_num][0])/float(table[w_num+G_num][0])))
            elif(peak_type == 'BWF'):
                rowID.append(float(table[A_num+D_num][0]))
                rowIG.append(float(table[A_num+G_num][0]))
                rowIDp.append(float(table[A_num+Dp_num][0]))
                rowI2D.append(float(table[A_num+TwoD_num][0]))           
                rowIDvsIG.append(float(table[A_num+D_num][0])/(float(table[A_num+G_num][0])))
                rowIDpvsIG.append(float(table[A_num+Dp_num][0])/(float(table[A_num+G_num][0])))
                rowI2DvsIG.append(float(table[A_num+TwoD_num][0])/(float(table[A_num+G_num][0])))
                
            rowPD.append(float(table[pos_num+D_num][0]))
            rowPG.append(float(table[pos_num+G_num][0]))
            rowPDp.append(float(table[pos_num+Dp_num][0]))
            rowP2D.append(float(table[pos_num+TwoD_num][0]))

            rowFWD.append(float(table[w_num+D_num][0]))
            rowFWG.append(float(table[w_num+G_num][0]))
            rowFWDp.append(float(table[w_num+Dp_num][0]))
            rowFW2D.append(float(table[w_num+TwoD_num][0]))

            try:
                doping_single,strain_single = calculateDopingEpsilon(float(table[pos_num+G_num][0]), float(table[pos_num+TwoD_num][0]), defVector0, defEpsilontocm1, defDopingtocm1, 
                       defchangeBaseMatrix)
            except:
                doping_single,strain_single = 0.0,0.0
            rowStrain.append(strain_single)
            rowDoping.append(doping_single)
                
        AD.append(rowAD)
        AG.append(rowAG)
        ADp.append(rowADp)
        A2D.append(rowA2D)
        
        ID.append(rowID)
        IG.append(rowIG)
        IDp.append(rowIDp)
        I2D.append(rowI2D)  
        
        IDvsIG.append(rowIDvsIG)
        IDpvsIG.append(rowIDpvsIG)
        I2DvsIG.append(rowI2DvsIG)
        
        P2D.append(rowP2D)
        PD.append(rowPD) 
        PDp.append(rowPDp)
        PG.append(rowPG)
        
        FW2D.append(rowFW2D)
        FWD.append(rowFWD)
        FWDp.append(rowFWDp)
        FWG.append(rowFWG)
        
        Strain.append(rowStrain)
        Doping.append(rowDoping)      
        
    #Save all the intensity values
    listofparameters = [AD,AG,ADp,A2D,
                        ID,IG,IDp,I2D,
                        IDvsIG,IDpvsIG,I2DvsIG,
                        PD,PG,PDp,P2D,
                        FWD,FWG,FWDp,FW2D,
                        Strain,Doping]
    #----Fix possible 'nan' values
    listofparameters_fixed = []
    for single_matrix in listofparameters:
        single_matrix = np.array(single_matrix) #Needed for nan stuff
        np.nan_to_num(single_matrix,False) # Fix for "not a number" values
        listofparameters_fixed.append(single_matrix)  
    listofparameters = listofparameters_fixed
    #----
    listofnames = ["AD","AG",'ADp',"A2D",
                   "ID","IG","IDp","I2D",
                   "IDvsIG","IDpvsIG","I2DvsIG",
                   "PD","PG","PDp","P2D",
                   "FWD","FWG","FWDp","FW2D",
                   "Strain","Doping"]
    saveallMatrixFiles(folderToLoad,listofparameters, listofnames)
    #Now the "clean" ones, after filtering
    graphenemask = getGrapheneMask(AG,FWD)
    listofparameters_filtered = []
    listofnames_filtered = []
    for i in listofparameters:
        listofparameters_filtered.append(i * graphenemask)
    for i in listofnames:
        listofnames_filtered.append(i + "filter")
    saveallMatrixFiles(folderToLoad,listofparameters_filtered, listofnames_filtered)
    
def fitMatrix(xspectra,spectraMatrix,filenameToSave = deffilenameToSave, 
              row_number = numberofrows, col_number = numberofcols, 
              peak_type = 'lorentz', remove_CosmicRays = True,
              remove_background = True, check_background = True):
    """Needs a /Matrix/ folder on the working directory
    peak_type: 'lorentz' or 'BWF'"""    
    if debug:
        print (xspectra,spectraMatrix[:,0*col_number+0+1])
    
    AD, AG, ADp, A2D = [], [], [], []
    ID, IG, IDp, I2D = [], [], [], []
    IDvsIG, IDpvsIG, I2DvsIG = [], [], []   
    P2D, PD, PDp, PG = [], [], [], []
    FW2D, FWD, FWDp, FWG = [], [], [], []
    Strain, Doping = [], []
    #numbers to easily change matrix positions
    peak_num = 7
    pos_num, w_num, A_num = [0*peak_num, 1*peak_num, 2*peak_num]
    D_num, G_num, Dp_num, TwoD_num = [0,1,2,4]
    
    pathtosave = os.path.join(filenameToSave,'Matrix')
    if not os.path.exists(pathtosave):
        os.makedirs(pathtosave)
    for i in range(col_number-1):
        rowAD, rowAG, rowADp, rowA2D = [], [], [], []
        rowID, rowIG, rowIDp, rowI2D = [], [], [], []
        rowIDvsIG, rowIDpvsIG, rowI2DvsIG = [], [], []
        rowP2D, rowPD, rowPDp, rowPG = [], [], [], []
        rowFW2D, rowFWD, rowFWDp, rowFWG = [], [], [], []
        rowStrain, rowDoping = [], []
        for j in range(row_number):
            print("Row,Col:", j, i, "Row in file:", j*col_number+i+1)
            fitted_values, fitted_cov = fitPeaks(xspectra, spectraMatrix[:,j*col_number+i+1],
                                                 os.path.join(pathtosave,"({0},{1})".format(i,j)),
                                                 peak_type = peak_type, 
                                                 remove_CosmicRays = remove_CosmicRays,
                                                 remove_background =remove_background, 
                                                 check_background = check_background)
            print("Fitted values:",fitted_values)
            rowAD.append(fitted_values[A_num+D_num])
            rowAG.append(fitted_values[A_num+G_num])
            rowADp.append(fitted_values[A_num+Dp_num])
            rowA2D.append(fitted_values[A_num+TwoD_num])
            
            if(peak_type == 'lorentz'):
                rowID.append(fitted_values[A_num+D_num]*2/(np.pi*fitted_values[w_num+D_num]))
                rowIG.append(fitted_values[A_num+G_num]*2/(np.pi*fitted_values[w_num+G_num]))
                rowIDp.append(fitted_values[A_num+Dp_num]*2/(np.pi*fitted_values[w_num+Dp_num]))
                rowI2D.append(fitted_values[A_num+TwoD_num]*2/(np.pi*fitted_values[w_num+TwoD_num])) 
                rowIDvsIG.append((fitted_values[A_num+D_num]/fitted_values[w_num+D_num])
                /(fitted_values[A_num+G_num]/fitted_values[w_num+G_num]))
                rowIDpvsIG.append((fitted_values[A_num+Dp_num]/fitted_values[w_num+Dp_num])
                /(fitted_values[A_num+G_num]/fitted_values[w_num+G_num]))
                rowI2DvsIG.append((fitted_values[A_num+TwoD_num]/fitted_values[w_num+TwoD_num])
                /(fitted_values[A_num+G_num]/fitted_values[w_num+G_num]))
 
            if(peak_type == 'BWF'):
                rowID.append(fitted_values[A_num+D_num])
                rowIG.append(fitted_values[A_num+G_num])
                rowIDp.append(fitted_values[A_num+Dp_num])
                rowI2D.append(fitted_values[A_num+TwoD_num]) 
                rowIDvsIG.append(fitted_values[A_num+D_num]/fitted_values[A_num+G_num])
                rowIDpvsIG.append(fitted_values[A_num+Dp_num]/fitted_values[A_num+G_num])
                rowI2DvsIG.append(fitted_values[A_num+TwoD_num]/fitted_values[A_num+G_num])
                
            rowPD.append(fitted_values[pos_num+D_num])
            rowPG.append(fitted_values[pos_num+G_num])
            rowPDp.append(fitted_values[pos_num+Dp_num])
            rowP2D.append(fitted_values[pos_num+TwoD_num])

            rowFWD.append(fitted_values[w_num+D_num])
            rowFWG.append(fitted_values[w_num+G_num])
            rowFWDp.append(fitted_values[w_num+Dp_num])
            rowFW2D.append(fitted_values[w_num+TwoD_num])
            
            try:
                doping_single,strain_single = calculateDopingEpsilon(fitted_values[pos_num+G_num], fitted_values[pos_num+TwoD_num], defVector0, defEpsilontocm1, defDopingtocm1, 
                       defchangeBaseMatrix)
            except:
                doping_single,strain_single = 0.0,0.0
            rowStrain.append(strain_single)
            rowDoping.append(doping_single)
            
        AD.append(rowAD)
        AG.append(rowAG)
        ADp.append(rowADp)
        A2D.append(rowA2D)        
        
        ID.append(rowID)
        IG.append(rowIG)
        IDp.append(rowIDp)
        I2D.append(rowI2D)
        
        IDvsIG.append(rowIDvsIG)
        IDpvsIG.append(rowIDpvsIG)
        I2DvsIG.append(rowI2DvsIG)
        
        P2D.append(rowP2D)
        PD.append(rowPD)
        PDp.append(rowPDp)
        PG.append(rowPG)
        
        FW2D.append(rowFW2D)
        FWD.append(rowFWD)
        FWDp.append(rowFWDp)
        FWG.append(rowFWG)
        
        Strain.append(rowStrain)
        Doping.append(rowDoping)
        
    #Save all the intensity values
    listofparameters = [AD,AG,ADp,A2D,
                        ID,IG,IDp,I2D,
                        IDvsIG,IDpvsIG,I2DvsIG,
                        PD,PG,PDp,P2D,
                        FWD,FWG,FWDp,FW2D,
                        Strain,Doping]
    #----Fix possible 'nan' values
    listofparameters_fixed = []
    for single_matrix in listofparameters:
        single_matrix = np.array(single_matrix) #Needed for nan stuff
        np.nan_to_num(single_matrix,False) # Fix for "not a number" values
        listofparameters_fixed.append(single_matrix)  
    listofparameters = listofparameters_fixed
    #----
    listofnames = ["AD","AG", 'ADp',"A2D",
                   "ID","IG","IDp","I2D",
                   "IDvsIG","IDpvsIG","I2DvsIG",
                   "PD","PG","PDp","P2D",
                   "FWD","FWG","FWDp","FW2D",
                   "Strain","Doping"]
    
    saveallMatrixFiles(pathtosave,listofparameters, listofnames)
    #Now the "clean" ones, after filtering
    graphenemask = getGrapheneMask(AG, FWG)
    listofparameters_filtered = []
    listofnames_filtered = []
    for i in listofparameters:
        listofparameters_filtered.append(i * graphenemask)
    for i in listofnames:
        listofnames_filtered.append(i + "filter")
    saveallMatrixFiles(pathtosave,listofparameters_filtered, listofnames_filtered)  




    
    
def Lanzarajuste_lineas(names):


    fichero, archivo = os.path.split(names) 
    print(names)
    filas = 1 # de 0 a filas-1 van de izquierda a derecha
    
    folder_to_save = r'Analysis'
    filewithmatrix = names
    spectraMatrix = np.loadtxt(filewithmatrix)
    columnas = spectraMatrix.shape[1] # de 0 a columnas-1 van de arriba a abajo
    
    ancho = 1 # en micras, lo de las columnas (de arriba a abajo de la pantalla, con las columnas)
    #Para hacer el ajuste a lorentzianas
    numberofcols = columnas
    numberofrows = filas
    peak_type = 'lorentz'
    RemoveCosmicRays = True
    RemoveBackground = True
    CheckRemoveBackground = True
    
    
    foldertosaveResults= os.path.join(folder_to_save,archivo[:-4])
    print(foldertosaveResults)
    # if not os.path.isdir(foldertosaveResults):
    #     os.mkdir(foldertosaveResults)
    # for i in spectraMatrix.shape[1]:
    #     x, y = spectraMatrix[:,0], spectraMatrix[:,:]
    x,y = loadMatrix(filewithmatrix)
    
    fitMatrix(x,y,foldertosaveResults,numberofrows,numberofcols,peak_type,
              RemoveCosmicRays,RemoveBackground,CheckRemoveBackground)

    # x,y = loadMatrix(filewithmatrix)
    # print ("Matrix loaded")
    # 
    # micron_per_px = ancho/numberofcols
    
def multiplefoldersfiles():
    directorytoopen= r'To_be_analyzed_lines' #ask for a folder with multiple subfolders to open

    for root,dirs,files in os.walk(directorytoopen):
        for name in files:
            names=os.path.join(root, name)
            if names.lower().endswith(('.txt')):
                Lanzarajuste_lineas(names)
                
                
if __name__ == '__main__':
    
    multiplefoldersfiles()