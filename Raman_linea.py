# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:55:04 2021

@author: msinloz


20220512: Cambio de nombre del archivo: Raman_lineas
20220513: Introduzco modificación de los standares de pintar modificando .rc
"""

filas = 1 # de 0 a filas-1 van de izquierda a derecha
columnas = 1 # de 0 a columnas-1 van de arriba a abajo
fichero = "//content//#024_021_mod.txt" # El archivo a subir a la izquierda en la pestaña "Archivo"
ancho = 1 # en micras, lo de las columnas (de arriba a abajo de la pantalla, con las columnas)

import numpy as np
import scipy.constants as phys_constants
import matplotlib.pyplot as plt

from matplotlib.pyplot import (plot,xlabel,ylabel,savefig,clf,title,figure,
                                imshow,contour,colorbar) 
import scipy.optimize as optimize
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import os, time
import re
import tkinter
from tkinter import filedialog



################################################

plt.rc('lines', linewidth=1., markersize=5)
plt.rc('grid', linewidth=0.5, ls='--', c='k')
plt.rc('xtick', direction='in',top='True',labelsize=10)
plt.rc('ytick', direction='in',right='True',labelsize=10)
plt.rc('font',family='serif')
# plt.rc('legend', numpoints=1,)


##################################################


"""
Created on Thu Jan 15 10:04:10 2015
Script for calculating strain and doping levels using
the calculations in 10.1038/ncomms2022
@author: Alberto
v2: Modified 17.07.2019 Stop using matrices, using arrays instead
"""
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
    print (xarray,yarray)
    plot(xarray,yarray,'r')
    
    
"""
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
@author: albosca"""

#Some initial variables
#class RamanConfiguration(matrixfile, folder, rows, cols, initialvalues):
#    """Class to save and load configuration info easily"""
#    pass
    
defFolder = '/home/albosca/Documentos/peak-o-mat DATA/matrix/'
deffileToLoad = '/home/albosca/Documentos/peak-o-mat DATA/7 picos.lpj'
deffolderToSave = '/home/albosca/Documentos/peak-o-mat DATA/'
deffilenameToSave = 'load'
defmatrixToLoad = '/home/albosca/Documentos/peak-o-mat DATA/XY 50X  LS30 Pol 0 grados_003_(Sub BG) (Sub BG).txt'
defspectrumToLoad = '/home/albosca/Documentos/peak-o-mat DATA/RamanTest2.dat'
posXfile = 0
posYfile = 0
debug = False  #verbose mode if true
saveToFile = True
numofiterations = 1000
xaxis_min = 1150 #cm-1
xaxis_max = 3500 #cm-1

#----Define model initial values---
#----For LORENTZ----
peakPosValues = [1349, 1594, 1630, 2480,
                 2689,2960, 3253]
peakPosBoundaries = [[1300,1390],[1500,1640],[1620,1650],[2400,2500],
                     [2610,2750],[2890,3000],[3100,3350]]
peakFwhmValues = [30, 30, 30, 30, 30, 30, 30] 
peakFwhmBoundaries = [[0.5,150], [0.5,200], [0.5,80], [0.5,150], ###Limita los valores de los fitting de cada pico// Cambio picos G y D' de 60 a 100, picos 2D
                      [0.5,200], [0.5,150], [0.5,150]]
#[[0.5,80], [0.5,100], [0.5,100], [0.5,80], ##Limita los valores de los fitting de cada pico// Cambio picos G y D' de 60 a 100, picos 2D
                      #[0.5,150], [0.5,80], [0.5,60]]

peakAmpValues = [100000, 100000, 100, 100, 100000, 100, 100]  #[100., 100., 0.1, 0.1, 100., 0.1, 0.1] 
peakAmpBoundaries = [[0.5,25000000.], [0.5,25000000.], [0.05,25000000.], [0.05,25000000.], #añado 1 0 a cada numero
                     [0.5,25000000.], [0.05,25000000.], [0.05,25000000.]]

#[[0.5,50000.], [0.5,50000.], [0.05,50000.], [0.05,50000.], 
         #            [0.5,50000.], [0.05,50000.], [0.05,50000.]]
#----End For LORENTZ----
# #----For BWF----
# peakPosValues = [1349, 1594, 1630, 2480,
#                 2689,2960, 3253]
# peakPosBoundaries = [[1300,1390],[1500,1640],[1620,1650],[2400,2530],
#                     [2610,2750],[2890,3000],[3100,3350]]
# peakFwhmValues = [1, 10, 1, 1, 1, 30, 1] 
# peakFwhmBoundaries = [[1,30], [2,60], [1,60], [1,80], 
#                       [1,110], [1,80], [1,60]]

# peakAmpValues = [5, 100., 0.1, 10, 100., 10, 10] 
# peakAmpBoundaries = [[0.05,500000.], [0.5,500000.], [0.05,500000.], [0.05,500000.], 
#                     [0.5,500000.], [0.05,50000.], [0.05,50000.]]

# peakqValues = [10000., 10000., 10000., 10000., 10000., 10000., 10000.] 
# peakqBoundaries = [[100.,10000000.],[100.,10000000.],[100.,10000000.],
#                   [100.,10000000.],[100.,10000000.],[100.,10000000.],
#                   [100.,10000000.]]
# #----End For BWF----


def_initialValues = peakPosValues
for i in peakFwhmValues:
    def_initialValues.append(i)
for i in peakAmpValues:
    def_initialValues.append(i)
#for i in peakqValues:
#    def_initialValues.append(i)

boundaries_low = []
boundaries_high = []
for i in peakPosBoundaries:
    boundaries_low.append(i[0])
    boundaries_high.append(i[1])
for i in peakFwhmBoundaries:
    boundaries_low.append(i[0])
    boundaries_high.append(i[1])
for i in peakAmpBoundaries:
    boundaries_low.append(i[0])
    boundaries_high.append(i[1])
#for i in peakqBoundaries:
#    boundaries_low.append(i[0])
#    boundaries_high.append(i[1])

boundaries_tuple = (boundaries_low,boundaries_high)

#----For matrix operations----
laserExcitation = 532.224 #nm
cmTonm = 1e7
numberofcols = 45
numberofrows = 45
defRemoveCosmicRays = True
defRemoveBackground = True
defCheckRemoveBackground = True
xaxis_in_nm = False
defIntervalBG = [800., 1200., 2000., 2200., 3100., 3500.]
defIntervalSize = 30
deffilter_gradient = 100 # For removing points with large derivative (cosmic ray spikes)

#----Strain and doping calculation (VectoresLee2012)----
defPosG0 = 1581.6 # in cm-1. For lambda = 514.5nm 1581.6 cm-1
defPos2D0 = 2669.9 # in cm-1. For lambda = 514.5nm 2676.9 cm-1
#lambda = 473nm 2695.5 cm-1 
#lambda = 532nm 2669.9 cm-1 
#Tensile strain values
defSlopeT = 2.2 #adim w2D/wG
#Hole doping values
defSlopeH = 0.70 #adim w2D/wG
#Conversion factors
defEpsilontocm1 = -69.1 # cm-1/% -23.5 uniaxial -69.1 biaxial
defDopingtocm1 = -1.04/1e12 # cm-1/n(cm-2)/
defVector0,defeT,defeH,defchangeBaseMatrix = generateUnitaryVectors(defPosG0, 
                                                                    defPos2D0,
                                                                    defSlopeT,
                                                                    defSlopeH)
#Usage example
#calculateEpsilonDoping(1590, 2700, defVector0, defEpsilontocm1, defDopingtocm1, 
#                       defchangeBaseMatrix)




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

def loadspectrum(spectrumToLoad = defspectrumToLoad):
    """Get x and y from one .dat file"""
    spectrumLoaded = np.loadtxt(spectrumToLoad)
    return spectrumLoaded.T[0], spectrumLoaded.T[1] 

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
 

def backgroundRemoval(x,y, bgpoints = defIntervalBG, bgsize = defIntervalSize,
                      plot_check = defCheckRemoveBackground, save_folder = deffolderToSave):
    """Needs the x and y to remove. Improved for removing spikes"""

    quadratic_func = lambda x,a,b,c: a+b*x+c*(x**2)
    #Create a mask to select only certain regions for BG
    mask = x>1e6 # Generate False array
    for i in bgpoints:
        mask = np.logical_or(mask,((i<x)*(x<i+bgsize))) # Select true only defined regions
    #Fit the curve
    fitted_val, fitted_cov = optimize.curve_fit(quadratic_func, x[mask], y[mask], p0 = [0.1,0.001,0.0001])
    print (fitted_val, fitted_cov)
    x_bg = x
    y_bg = quadratic_func(x,*fitted_val)
    if plot_check:
        fig = figure(1)
        fig.clf()
        axes = fig.add_subplot(111)
        axes.set_xlabel("cm-1")
        axes.set_ylabel("Intensity (a.u)")
        axes.set_title('Check Background removal')
        axes.set_xlim(xaxis_min,xaxis_max)
        #Autoscale the y axis in the range
        y_limited = y[(xaxis_min<x)*(x<xaxis_max)]
        axes.set_ylim(y_limited.min(),y_limited.max())
        axes.plot(x, y, 'go')
        axes.plot(x_bg, y_bg, 'r-')
        axes.autoscale_view(tight=None, scalex=False, scaley=True)
        savefig(save_folder+'_bg.svg',dpi = 500)
        
    return x, y-y_bg

def multiLorentzians(x, position, fwhm_simple, amp):
    """Fit to any number of lorenzians"""
    if (len(position)!=len(fwhm_simple) or len(position)!=len(amp) or len(amp)!=len(fwhm_simple)):
        print("Invalid number of parameters")
        return 0
    ypeaks = 0
    for i, pos in enumerate(position):
        ypeaks += 2*amp[i]*fwhm_simple[i]/(np.pi*(4*((x-pos)**2)+fwhm_simple[i]**2))#Same as in origin and witec program
        # print(ypeaks)
    return ypeaks

def sevenLorentzians(x,p0,p1,p2,p3,p4,p5,p6,w0,w1,w2,w3,w4,w5,w6,a0,a1,a2,a3,a4,a5,a6):
    """Wrapper for multilorenzians"""
    return multiLorentzians(x,[p0,p1,p2,p3,p4,p5,p6],
                            [w0,w1,w2,w3,w4,w5,w6],
                            [a0,a1,a2,a3,a4,a5,a6])

def multiBWF(x, position, fwhm_simple, amp, q_asymmetry):
    """Fit to any number of BWF peaks"""
    if (len(position)!=len(fwhm_simple) or len(position)!=len(amp) or 
        len(amp)!=len(fwhm_simple) or len(position)!=len(q_asymmetry)):
        print("Invalid number of parameters")
        return 0
    ypeaks = 0
    for i, pos in enumerate(position):
        ypeaks += amp[i]*((1+(x-pos)/(q_asymmetry[i]*fwhm_simple[i]))**2)/(1+(((x-pos)/fwhm_simple[i])**2)) #Same as in origin and witec program
    return ypeaks

def sevenBWF(x,p0,p1,p2,p3,p4,p5,p6,
             w0,w1,w2,w3,w4,w5,w6,
             a0,a1,a2,a3,a4,a5,a6,
             q0,q1,q2,q3,q4,q5,q6):
    """Wrapper for multiBWF"""
    return multiBWF(x,[p0,p1,p2,p3,p4,p5,p6],
                    [w0,w1,w2,w3,w4,w5,w6],
                    [a0,a1,a2,a3,a4,a5,a6],
                    [q0,q1,q2,q3,q4,q5,q6])

    
def fitPeaks(x, y, filenameToSave = deffilenameToSave, peak_type = 'lorentz',
             peak_number = 7, initial_values = def_initialValues, 
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
        x,y = backgroundRemoval(x,y,defIntervalBG,defIntervalSize,
                                check_background,filenameToSave)
    #----Define plot to use--
    fig = figure(1)
    fig.clf()
    axes = fig.add_subplot(111)
    axes.set_xlabel("cm-1")
    axes.set_ylabel("Intensity (a.u)")
    axes.set_title('Raman peak fitting')
    axes.set_xlim(xaxis_min,xaxis_max)
    #Autoscale the y axis in the range
    y_limited = y[(xaxis_min<x)*(x<xaxis_max)]
    axes.set_ylim(y_limited.min(),y_limited.max())
    axes.plot(x, y, 'g-')
    axes.autoscale_view(tight=None, scalex=False, scaley=True)

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
        elif (peak_type == 'BWF'):
            fitted_values, fitted_cov = optimize.curve_fit(sevenBWF, x, y, bounds = boundaries, p0 = initial_values)
            ymodel = sevenBWF(xmodel, *fitted_values)
            fitted_error = np.sqrt(np.diag(fitted_cov))
            y_limited = np.append(ymodel[(xaxis_min<x)*(x<xaxis_max)],y[(xaxis_min<x)*(x<xaxis_max)]) # resize y axis
    
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
    
    axes.set_ylim(y_limited.min(),y_limited.max())    #resize y axis
    axes.plot(xmodel,ymodel,'r-')
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
        axes.plot(xmodel,ymodelpeak,'b-')
    
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
        savefig(filenameToSave + '.svg',dpi = 400)
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

def createGrapheneMaskFile(folderToLoad):
    """Save text file with true/false graphene mask"""
    if os.path.isdir(folderToLoad):
        filenames = os.listdir(folderToLoad)  
    else:
        return 0
    AGfile= ""
    FWGfile= ""
    for i in filenames:
        if (i == 'AG.mtx'):
            AGfile = folderToLoad+i
            AG = np.loadtxt(AGfile)  
        if (i == 'FWG.mtx'):
            FWGfile = folderToLoad+i
            FWG = np.loadtxt(FWGfile)  
    
    isgraphenemask = getGrapheneMask(AG,FWG)
    np.savetxt(folderToLoad+"grapheneMask.mtx",isgraphenemask)
    return AG,FWG,isgraphenemask

def distanceToCenter(matrix, center):
    num_of_rows = matrix[:][0].size
    num_of_cols = matrix[0][:].size
    distance = []
    value = []
    if((center[0]>num_of_rows-1) or (center[1]>num_of_cols-1)):
        raise("Bad center")
    for i in range(num_of_rows):
        for j in range(num_of_cols):
            distance.append(np.sqrt((center[0]-i)**2+(center[1]-j)**2))
            value.append(matrix[i][j])
    data_to_return = np.array([distance,value])
    return data_to_return  

def calculate_value_distance_center(matrix_to_plot, AG, FWG, num_of_sigmas = 3):
    """ Function to calculate the mean values of matrix_to_plot depending on the radius, in pixels
    IG,FWG and num_of_sigmas are needed to select graphene regions and calculate the grain center"""
    #get center
    center_grain = getCenter(AG, FWG, num_of_sigmas)
    #get distribution in terms of pixel distance
    distribution_distance = distanceToCenter(matrix_to_plot, center_grain)
    #get where we have graphene
    gr_mask_linear = getGrapheneMask(AG, FWG, num_of_sigmas).flatten()
    distance_gr = gr_mask_linear * distribution_distance[0]
    value_gr = gr_mask_linear * distribution_distance[1]
    distance_binning = np.linspace(distance_gr.min(), distance_gr.max(), int(distance_gr.size/200.))
    value_binning = []
    freq_binning = []
    for i in range(distance_binning.size-1):
        mask_condition = (distance_gr > distance_binning[i]) * (distance_gr <= distance_binning[i+1])
        value_binning.append(value_gr[mask_condition].mean())
        freq_binning.append(value_gr[mask_condition].size)
    value_binning = np.array(value_binning)
    freq_binning = np.array(freq_binning)
    return distance_binning[:-1], value_binning, freq_binning

def plot_value_distance_center(matrix_path, AG_path, FWG_path, time_seconds, micron_per_px = 1, num_of_sigmas = 3):
    matrix_to_plot =np.loadtxt(matrix_path)
    AG = np.loadtxt(AG_path)
    FWG = np.loadtxt(FWG_path)
    x,y,freq = calculate_value_distance_center(matrix_to_plot, AG, FWG, num_of_sigmas)
    if(micron_per_px !=1):
       x = micron_per_px * x 
    figure(1)
    clf()
    title("Pixel distribution")
    xlabel("Radius (px)")
    if(micron_per_px !=1):
        xlabel("Radius (µm)")
    ylabel("Binned number of points")
    plot(x,freq)
    np.savetxt(matrix_path[:-4]+'_r_freq.dat',np.c_[x,freq])
    savefig(matrix_path[:-4]+'_r_freq.svg')
    figure(2)
    clf()
    title("Radius vs. Matrix quantity")
    xlabel("Radius (px)")
    if(micron_per_px !=1):
        xlabel("Radius (µm)")
    ylabel("input matrix binned mean values")
    plot(x,y)
    np.savetxt(matrix_path[:-4]+'_r_val.dat',np.c_[x,y])
    savefig(matrix_path[:-4]+'_r_val.svg')
    figure(3)
    clf()
    title("Time vs. Matrix quantity")
    xlabel("time (s)")
    ylabel("input matrix binned mean values")
    k = time_seconds / (np.pi*x[-1]**2)
    t = k*np.pi *x**2
    plot(t,y)
    np.savetxt(matrix_path[:-4]+'_t_val.dat',np.c_[t,y])
    savefig(matrix_path[:-4]+'_t_val.svg')
    figure(4)
    clf()
    title("Radius vs. time")
    xlabel("Radius(px)")
    ylabel("time (s)")
    if(micron_per_px !=1):
        xlabel("Radius (µm)")
    plot(x,t)
    np.savetxt(matrix_path[:-4]+'_r_t.dat',np.c_[x,t])
    savefig(matrix_path[:-4]+'_r_t.svg')

def plot_value_distance_center_improved(matrix_path, AG_path, FWG_path, path_to_save, tau_chamber,
                                        time_growth, micron_per_px = 1, 
                                        num_of_sigmas = 3, empty_factor = 1):
    """23.09.2020 New model for the connection between radius and time
    rho(t)= int(v0*(1-exp(-t/tau)),dt)  t< tgrowth
            rho(tgrowth)int(v0*exp(-t/tau),dt)  """ 
    #Load the matrix and calculate the center
    matrix_to_plot =np.loadtxt(matrix_path)
    AG = np.loadtxt(AG_path)
    FWG = np.loadtxt(FWG_path)
    x,y,freq = calculate_value_distance_center(matrix_to_plot, AG, FWG, num_of_sigmas)
    if(micron_per_px !=1):
       x = micron_per_px * x 
    #Plot the matrix versus radius
    figure(1)
    clf()
    title("Pixel distribution")
    xlabel("Radius (px)")
    if(micron_per_px !=1):
        xlabel("Radius (µm)")
    ylabel("Binned number of points")
    plot(x,freq)
    np.savetxt(matrix_path[:-4]+'_r_freq.dat',np.c_[x,freq])
    savefig(matrix_path[:-4]+'_r_freq.svg')
    figure(2)
    clf()
    title("Radius vs. Matrix quantity")
    xlabel("Radius (px)")
    if(micron_per_px !=1):
        xlabel("Radius (µm)")
    ylabel("input matrix binned mean values")
    plot(x,y)
    np.savetxt(matrix_path[:-4]+'_r_val.dat',np.c_[x,y])
    savefig(matrix_path[:-4]+'_r_val.svg')

    #NEW as of 23.09.2020
    #Refined time calculation
    radius_for_max_value = x[y==y.max()] # get end of CH4 income
    print("max radius: {0} um".format(radius_for_max_value))
    mean_velocity = radius_for_max_value/integrate.quad(lambda t: (1.0-np.exp(-t/tau_chamber)),0,time_growth)[0]
    print("Mean velocity: {0} um/s".format(mean_velocity))
    t_array_growth = np.linspace(0,time_growth-0.1,200)
    t_array_stop = np.linspace(time_growth,time_growth+tau_chamber*10,200)
    # The integration must be onoe by one
    radius_from_t_growth = []
    for t_end in t_array_growth:
      radius_from_t_growth.append(integrate.quad(lambda t: mean_velocity*(1.0-np.exp(-t/tau_chamber)),0,t_end)[0])
    radius_from_t_growth =np.array(radius_from_t_growth)

    radius_from_t_stop = []
    for t_end in t_array_stop:
      radius_from_t_stop.append(radius_for_max_value + integrate.quad(lambda t:  mean_velocity*(np.exp(-(t-time_growth)/(tau_chamber*empty_factor))),time_growth,t_end)[0])
    radius_from_t_stop =np.array(radius_from_t_stop)

    #Now merge the two calculations and create a spline
    full_t_array = np.concatenate((t_array_growth,t_array_stop), axis=None)
    full_radius_array = np.concatenate((radius_from_t_growth,radius_from_t_stop), axis=None)
    print("t array:")
    print(full_t_array)
    print("radius array:")
    print(full_radius_array)
    time_from_rho_function = interp1d(full_radius_array, full_t_array,kind='cubic')
    print("rho to interpolate array:")
    print(x)
    print("t interpolated array:")
    print(time_from_rho_function(x))

    figure(3)
    clf()
    title("Time vs. Matrix quantity")
    xlabel("time (s)")
    ylabel("input matrix binned mean values")
    t = time_from_rho_function(x) #easy peasy using the interpolation
    plot(t,y)
    np.savetxt(path_to_save+'_t_val.dat',np.c_[t,y])
    savefig(path_to_save+'_t_val.svg')
    figure(4)
    clf()
    title("Radius vs. time")
    xlabel("Radius(px)")
    ylabel("time (s)")
    if(micron_per_px !=1):
        xlabel("Radius (µm)")
    plot(x,t)
    np.savetxt(path_to_save + '_r_t.dat',np.c_[x,t])
    np.savetxt(path_to_save + '_r_t_full.dat',np.c_[full_radius_array,full_t_array])
    savefig(path_to_save + '_r_t.svg')

      
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

def getMatrixAndStatistics(pathtoMatrix, remove_zeros = True):
    """Function to load a matrix and return it as a known array.
    Also, it prints statistical values"""
    matrix_array = np.loadtxt(pathtoMatrix)
    if(remove_zeros):
        print("Mean value:",matrix_array[matrix_array!=0].mean())
        print("Standard deviation:",matrix_array[matrix_array!=0].std())
    else:
        print("Mean value:",matrix_array.mean())
        print("Standard deviation:",matrix_array.std())
    return matrix_array

def convertForOrigin(matrix_list, output_filename, file_header= "\n"):
    """Function to put in one file all the info from the matrix list"""
    flat_matrix_list = []
    for single_matrix in matrix_list:
        flat_matrix_list.append(single_matrix.flatten())
    np.savetxt(output_filename,np.array(flat_matrix_list).T, 
               header= file_header, comments='') 
    
def Lanzarajuste():
    name= tkinter.filedialog.askopenfilename()
    fichero, archivo = os.path.split(name) 
    print(name)
    filas = 1 # de 0 a filas-1 van de izquierda a derecha
    columnas = 1 # de 0 a columnas-1 van de arriba a abajo
    
    ancho = 1 # en micras, lo de las columnas (de arriba a abajo de la pantalla, con las columnas)
    #Para hacer el ajuste a lorentzianas
    numberofcols = columnas
    numberofrows = filas
    peak_type = 'lorentz'
    RemoveCosmicRays = True
    RemoveBackground = True
    CheckRemoveBackground = True
    filewithmatrix = name
    
    foldertosaveResults= os.path.join(fichero,'Matrix')
    if not os.path.isdir(foldertosaveResults):
      os.mkdir(foldertosaveResults)
    x,y = loadMatrix(filewithmatrix)
    print ("Matrix loaded")
    fitMatrix(x,y,foldertosaveResults,numberofrows,numberofcols,peak_type,
              RemoveCosmicRays,RemoveBackground,CheckRemoveBackground)
    micron_per_px = ancho/numberofcols


def Lanzarajuste_lineas(names):


    fichero, archivo = os.path.split(names) 
    print(names)
    filas = 1 # de 0 a filas-1 van de izquierda a derecha
    
    
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
    
    
    foldertosaveResults= os.path.join(fichero,archivo[:-4])
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
    directorytoopen= tkinter.filedialog.askdirectory()#ask for a folder with multiple subfolders to open
    t=time.time()
    i=0
    for root,dirs,files in os.walk(directorytoopen):
        for name in files:
            names=os.path.join(root, name)
            if names.lower().endswith(('.txt')):
                Lanzarajuste_lineas(names)
                
                
#            print(names)   
    # print('Done! Files openned: ' , i, 'in', time.time()-t , 'seconds. Time per file:',(time.time()-t)/i)        
#----
errmsg = 'Error!'
tkinter.Button(text='Select any Witec file', command=multiplefoldersfiles, bg='yellow',height = 25, width = 50).pack()
tkinter.mainloop()  