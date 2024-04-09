# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:51:44 2023

@author: msinloz

Es igual que el de un archivo, salvo que entra en todos los archivos que haya en la carpeta

https://stackoverflow.com/questions/29156532/python-baseline-correction-library
"""

from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve
import numpy as np
from numpy.linalg import norm
import scipy.optimize as optimize

from lmfit import Model
import pandas as pd
import os
import matplotlib.pyplot as plt


plt.rc('lines', linewidth=1., markersize=6)
plt.rc('grid', linewidth=0.5, ls='--', c='k', alpha=0.5)
plt.rc('xtick', direction='in',top='True',labelsize=10)
plt.rc('ytick', direction='in',right='True',labelsize=10)
plt.rc('font',family='serif')
plt.rc('legend', numpoints=1,)

plt.rc('axes', axisbelow=True) #https://stackoverflow.com/questions/1726391/matplotlib-draw-grid-lines-behind-other-graph-elements





folder = r'C:/Users/msinloz/OneDrive - UPV/Python_Projects/2023_Javi/Raman_files'



def Open_file(file):
    

    read_xy_file = np.loadtxt (file, delimiter ='	')
    x = read_xy_file[:,0]
    y = read_xy_file[:,1]
    # print(x)
    return x, y

def baseline_als_optimized(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

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
    
    
    
#Now fitting sobre el resultado d, de la baselinaralps


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

def oneLorentzians(x, p0,
                      w0,
                      a0):
    """Wrapper for multilorenzians"""
    return multiLorentzians(x,[p0],
                            [w0],
                            [a0])
def twoLorentzians(x, p0, p1,
                      w0, w1,
                      a0, a1):
    """Wrapper for multilorenzians"""
    return multiLorentzians(x,[p0, p1],
                            [w0, w1],
                            [a0, a1])
#----Define model initial values---
#----For LORENTZ----



def pos_boundaries(peak_number=0):
    if peak_number == 0:
        p0 = 521.
        p0_Boundary = [515., 525.]
        
        peakPosValues = [p0]
        peakPosBoundaries = [p0_Boundary]
                       
        FW0 = 5.
        FW0_Boundary = [0.5, 20.]
                
        peakFwhmValues = [FW0] 
        peakFwhmBoundaries = [FW0_Boundary]
        
        A0 = 500.
        A0_Boundary = [100., 10000.]
        
        peakAmpValues = [A0] 
        peakAmpBoundaries = [A0_Boundary]
        
        def_initialValues = peakPosValues
        for i in peakFwhmValues:
            def_initialValues.append(i)
        for i in peakAmpValues:
            def_initialValues.append(i)

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

        
        boundaries_tuple = (boundaries_low,boundaries_high)
        
    else:
        p0 = 515.
        p0_Boundary = [508., 520.]
        
        p1 = 521.
        p1_Boundary = [515., 525.]
        
                
        peakPosValues = [p0, p1]
        peakPosBoundaries = [p0_Boundary, p1_Boundary]
                      
        FW0 = 5.
        FW0_Boundary = [0.5, 20.]
        
        FW1 = 5.
        FW1_Boundary = [0.5, 20.]
        
        peakFwhmValues = [FW0, FW1 ] 
        peakFwhmBoundaries = [FW0_Boundary, FW1_Boundary]
        
        A0 = 500.
        A0_Boundary = [100., 10000.]
        
        A1 = 1000.
        A1_Boundary = [100., 10000.]
        
        peakAmpValues = [A0, A1]  
        peakAmpBoundaries = [A0_Boundary, A1_Boundary]

        def_initialValues = peakPosValues
        for i in peakFwhmValues:
            def_initialValues.append(i)
        for i in peakAmpValues:
            def_initialValues.append(i)

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
        
        boundaries_tuple = (boundaries_low,boundaries_high)
        
    return def_initialValues, boundaries_tuple
# print(len(boundaries_high))
#Se crea la lista para poder guardar en una lista todos los fit


def columns_list(peak_number=1):
    
    columns = []
    # columns.insert(0, 'Filename')
    
    for i in range (peak_number):
                
        columns.append('P_{number}'.format(number= i))
                               
    for i in range (peak_number):
                    
        columns.append('FW_{number}'.format(number= i))
                                
    for i in range (peak_number):
                       
        columns.append('A_{number}'.format(number= i))

    
    return columns
        
#Ya definidas las funciones para poder hacer los fitting a partir de ahora todo lo que se hará será llamar a los archivos
#quitar los background y fittear





if __name__ == '__main__':
    
    #Comenzamos el FOR para poder hacer la iteración en todos los archivos 
   count = 0  #Contador para inicializar el primer dataframe con los resultados
   for root, dir, files in os.walk(folder):
       for file in files:
           if file.endswith('.txt'):
               print(os.path.join(root,file))
               file_absolute_path = os.path.join(root,file)
               
               x, y = Open_file(file_absolute_path)
       
               z, d, info = baseline_arPLS(y, ratio=1e-6, lam=1e4, niter=100, full_output=True)
       
               k = baseline_als_optimized(y, 1e4, 0.001, niter=10)
       
               p = y-k
               fig, (ax) = plt.subplots(1, 1)
               fig.set_size_inches(6.692913385826771, 4.136447244094488)
               ax.plot(x,y)
               ax.plot(x,z, color= 'r')
               ax.plot(x,d, color= 'k')
               
               ax.plot(x, k, color = 'g')
               ax.plot(x, p, color ='b')
               
               
               fig.savefig(file_absolute_path[:-4]+'_Background_removal.svg', format='svg')
               
               plt.close()
       
               try:
                   

                   initial_values, boundaries_tuple = pos_boundaries(peak_number=0)
                   fitted_values, fitted_cov = optimize.curve_fit(oneLorentzians, x, d, bounds = boundaries_tuple, p0=initial_values)
                   ymodel = oneLorentzians(x,*fitted_values)
                   perr = np.sqrt(np.diag(fitted_cov))
                   print(fitted_cov)
                   
                   if  np.mean(perr) < 15:          
                       #Print los valores del fitting en la consola  
                       columns = columns_list(peak_number=1)
                       dict_with_values = {column_name: [fitted] for column_name, fitted in zip(columns, fitted_values)}
                       dict_with_values['Filename'] = file
                       dict_with_values['perr'] = np.mean(perr)
                       # print(dictionary)
                       peak_number = 1
                       for i in range (peak_number):
                           print('P_{number}: {fitted}'.format(number= i, fitted = fitted_values[i]))
                           
                           print('FW_{number}: {fitted}'.format(number= i, fitted = fitted_values[i+peak_number]))
                           
                           print('A_{number}: {fitted}'.format(number= i, fitted = fitted_values[i+2*peak_number]))
                           
            
                       #Pinta la figura del fitting
                       
                       fig, (ax) = plt.subplots(1, 1)
                       fig.set_size_inches(6.692913385826771, 4.136447244094488)
                       
                       
                       ax.plot(x,d, color= 'k')
                       ax.plot(x,ymodel, color= 'y')
                       
                       for i in range(peak_number):
                    #        figure(2) # To check only the peaks
                           
                                ymodelpeak = multiLorentzians(x,[fitted_values[i]],
                                                              [fitted_values[i+peak_number]],
                                                              [fitted_values[i+2*peak_number]])
                                ax.plot(x,ymodelpeak)
                                
                       # print (fitted_values, fitted_cov)
                       ax.set_xlim(400, 600)
                       
                       plt.show()
                       
                       fig.savefig(file_absolute_path[:-4]+'_Fit.svg', format='svg')
                       
                       plt.close()
               
                       if count == 0:
                           df = pd.DataFrame(dict_with_values)
                           
                           count += 1
                       else:
                           df1= pd.DataFrame(dict_with_values)
                           df = pd.concat([df,df1])
                        # fitted_values.insert(0, fichero)
                       # df1 = pd.DataFrame(fitted_values_list)
                       # print(df1)
                       # df = pd.concat([df,df1], axis=1)
                       
                   else:
                       
                        initial_values, boundaries_tuple = pos_boundaries(peak_number=1)
                        fitted_values, fitted_cov = optimize.curve_fit(twoLorentzians, x, d, bounds = boundaries_tuple, p0=initial_values)
                        ymodel = twoLorentzians(x,*fitted_values)
                        perr = np.sqrt(np.diag(fitted_cov))
                        
                        columns = columns_list(peak_number=2)
                        dict_with_values = {column_name: [fitted] for column_name, fitted in zip(columns, fitted_values)}
                        
                        dict_with_values['Filename'] = file
                        dict_with_values['perr'] = np.mean(perr)

                        
                        
                        peak_number=2
                        for i in range (peak_number):
                            print('P_{number}: {fitted}'.format(number= i, fitted = fitted_values[i]))
                            
                            print('FW_{number}: {fitted}'.format(number= i, fitted = fitted_values[i+peak_number]))
                            
                            print('A_{number}: {fitted}'.format(number= i, fitted = fitted_values[i+2*peak_number]))
                            
             
                        #Pinta la figura del fitting
                        
                        fig, (ax) = plt.subplots(1, 1)
                        fig.set_size_inches(6.692913385826771, 4.136447244094488)
                        
                        
                        ax.plot(x,d, color= 'k')
                        ax.plot(x,ymodel, color= 'y')
                        
                        for i in range(peak_number):
                     #        figure(2) # To check only the peaks
                            
                                 ymodelpeak = multiLorentzians(x,[fitted_values[i]],
                                                               [fitted_values[i+peak_number]],
                                                               [fitted_values[i+2*peak_number]])
                                 ax.plot(x,ymodelpeak)
                                 
                        # print (fitted_values, fitted_cov)
                        ax.set_xlim(400, 600)
                        
                        plt.show()
                        
                        fig.savefig(file_absolute_path[:-4]+'_Fit.svg', format='svg')
                        
                        plt.close()
                
                
                
                        if count == 0:
                            df = pd.DataFrame(dict_with_values)
                            count += 1
                        else:
                            df1= pd.DataFrame(dict_with_values)
                            df = pd.concat([df,df1])
                            
                            
                        #Pasa los valores fitteados en una lista y va añadiendolos a un dataframe para posteriormente guardarlo en el csv
                        # fitted_values_list=fitted_values.tolist()
                        # fitted_values_list.insert(0,file)
                        # fitted_values_list.append(np.mean(perr))
                        # print(fitted_values_list)
                         # fitted_values= np.insert(fitted_values, 0, 'a')
                         # print(fitted_values)
                         # print(len(columns), len(fitted_values))
                        
                         # fitted_values.insert(0, fichero)
                        # df1 = pd.DataFrame(fitted_values_list)
                        # print(df1)
                        # df = pd.concat([df,df1], axis=1)
                # print(df.T)
               finally:
                    df.to_csv('Fitting.csv')
                    print('FIN')
              