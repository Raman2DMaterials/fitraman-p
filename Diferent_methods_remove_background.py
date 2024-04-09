# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:11:53 2023

@author: msinloz

Preuba comparativa de los diferentes métodos de remover el bakground de medidas de Raman
https://stackoverflow.com/questions/29156532/python-baseline-correction-library
"""


from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve
import numpy as np
from numpy.linalg import norm
import scipy.optimize as optimize
import csv
# from lmfit import Model
import pandas as pd
import os
import matplotlib.pyplot as plt


################################################

plt.rc('lines', linewidth=1., markersize=5)
plt.rc('grid', linewidth=0.5, ls='--', c='k')
plt.rc('xtick', direction='in',top='True',labelsize=10)
plt.rc('ytick', direction='in',right='True',labelsize=10)
plt.rc('font',family='serif')
# plt.rc('legend', numpoints=1,)


##################################################



folder = r'Test_background'
######Párámetros necesarios bakground removal


#############################

defRemoveCosmicRays = True
defRemoveBackground = True
defCheckRemoveBackground = True
xaxis_in_nm = False

defIntervalBG = [800., 1200., 2000., 2200., 3100., 3500.]
defIntervalSize = 30
deffilter_gradient = 100 # For removing points with large derivative (cosmic ray spikes)



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
    
def backgroundRemoval(x,y, bgpoints = defIntervalBG, bgsize = defIntervalSize,
                      plot_check = defCheckRemoveBackground, save_folder = folder):
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
    # if plot_check:
    #     fig = figure(1)
    #     fig.clf()
    #     axes = fig.add_subplot(111)
    #     axes.set_xlabel("cm-1")
    #     axes.set_ylabel("Intensity (a.u)")
    #     axes.set_title('Check Background removal')
    #     axes.set_xlim(xaxis_min,xaxis_max)
    #     #Autoscale the y axis in the range
    #     y_limited = y[(xaxis_min<x)*(x<xaxis_max)]
    #     axes.set_ylim(y_limited.min(),y_limited.max())
    #     axes.plot(x, y, 'go')
    #     axes.plot(x_bg, y_bg, 'r-')
    #     axes.autoscale_view(tight=None, scalex=False, scaley=True)
    #     savefig(save_folder+'_bg.svg',dpi = 500)
        
    return y_bg, y-y_bg    

if __name__ == '__main__':
    
    #Comenzamos el FOR para poder hacer la iteración en todos los archivos 
   count = 0  #Contador para inicializar el primer dataframe con los resultados
   for root, dir, files in os.walk(folder):
       for file in files:
           if file.endswith('.txt'):
               print(os.path.join(root,file))
               file_absolute_path = os.path.join(root,file)
               
               x, y = Open_file(file_absolute_path)
       
               z, d, info = baseline_arPLS(y, ratio=1e-6, lam=1e5, niter=100, full_output=True)
               print(info)
               k = baseline_als_optimized(y, 1e6, 0.001, niter=100)
               
               
               y_bg, y_backremoved = backgroundRemoval(x,y, bgpoints = defIntervalBG, bgsize = defIntervalSize,
                                     plot_check = defCheckRemoveBackground, save_folder = folder)
               
               p = y-k
               fig, (ax) = plt.subplots(1, 1)
               fig.set_size_inches(6.692913385826771, 4.136447244094488)
               ax.plot(x,y, '--')
               ax.plot(x,z, color= 'r', label = 'arPLS')
               ax.plot(x,d, color= 'k', label = 'arPLS')
               
               ax.plot(x, k, color = 'g', label = 'als')
               ax.plot(x, p, color ='b', label = 'als')
               
               ax.plot(x, y_bg, color = 'y', label = 'bkg')
               ax.plot(x, y_backremoved, color ='c', label= 'bkg')
               
               ax.set_xlim(100,3500)
               ax.set_ylim(-1000,20000) 
               ax.legend()
               fig.savefig(file_absolute_path[:-4]+'_Background_removal.svg', format='svg')
               
               plt.close()

               with open(file[:-4]+'fYFWHM.csv','a') as f:

                wrt=csv.writer(f,delimiter=';')
            
                to_save = zip(x, y, z, d)
                for row in to_save:
                    wrt.writerow(row)
                
                f.closed