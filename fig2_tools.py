#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 01:49:15 2021

@author: dekens
"""


import numpy as np
import os
import matplotlib as ml
import matplotlib.pyplot as plt
from matplotlib import cm
from multiprocessing import Pool
from itertools import repeat
from scipy import sparse as sp
import scipy.sparse.linalg as scisplin


#### First function called by main code
#### Run run_selection in parallel (per selection function)
def run_Selection(Mfun, Mfun_string, Iinf, Isup, parameters, workdir):
    p  = Pool(3)
    inputs = [*zip(Mfun, Mfun_string, repeat(Iinf), repeat(Isup), repeat(parameters), repeat(workdir))]
    p.starmap(run_selection, inputs)

#### Second function, called by the previous one (run_Selection)
#### Save the final allelic distributions n and displays the marginals' temporal dynamics (rho_X and rho_Y)
#### Calls the function update, which computes the allelic distribution at the next time step

def run_selection(mfun, mfun_string, I_inf, I_sup, parameters, workdir):
        subtitle = 'selection_' + mfun_string +'space_%4.1f' %I_inf + '_%4.1f' %I_sup
        
        ######### Parameters
        eps, Tmax, Nt, Nx, Ny, i0, j0, i1, j1, kappa, r, g = parameters
        ## Time
        T, dt = np.linspace(0, Tmax, Nt, retstep = True)
        
        ## Allelic traits
        X, dx = np.linspace(I_inf, I_sup, Nx, retstep = True)
        Y, dy = np.linspace(I_inf, I_sup, Ny, retstep = True)
        
        ## Grid points for the initial state
        x0, y0 = X[i0], Y[j0]
        x1, y1 = X[i1], Y[j1]
        
        ### Creating directory 
        subworkdir = workdir+'/'+subtitle+'_x0 = %4.2f'%x0+'_x1 = %4.2f'%x1+'_y0 = %4.2f'%y0+'_y1 = %4.2f'%y1
        create_directory(subworkdir)
        
        #### create local directory
        local_dir = subworkdir+'/n_e_log10(eps) =%i'%np.floor(np.log10(eps))+'_Tmax = %i'%Tmax+'_dt = %4.3f'%dt
        create_directory(local_dir)
        
        #### Loop to create matrix with selection values over allelic space
        M = np.zeros((Nx, Ny))
        for j in range(Ny):
            for i in range(Nx):
                M[i, j] = g*mfun(X[i], Y[j])
        m = mat_to_vect(M, Nx, Ny)
        Id = sp.spdiags(np.ones(Nx*Ny), 0, Nx*Ny, Nx*Ny)
        
        ######## Initial state n0
        
        first_dirac = r/(kappa*2) * ( gaussian_2d(X, Y, x0, y0, np.sqrt(eps), np.sqrt(eps)))
        second_dirac = r/(kappa*2) * ( gaussian_2d(X, Y, x1, y1, np.sqrt(eps), np.sqrt(eps)))
        n0 = first_dirac + second_dirac
                
        #### Variables of interest
        
        Rho = np.zeros(Nt)
        Rho_X = np.zeros((Ny, Nt))
        Rho_Y = np.zeros((Nx, Nt))
        Rho[0], Rho_X[:, 0], Rho_Y[:, 0] = np.sum(n0)*dx*dy, np.sum(n0, axis = 0)*dx, np.sum(n0, axis = 1)*dy
        n = 10*n0
        
        ##### Run iterations of numerical scheme
        for nt in range(Nt-1):
            n, Rho[nt+1], Rho_X[:, nt+1], Rho_Y[:, nt+1] = update(n, Rho[nt], Rho_X[:, nt], Rho_Y[:, nt], parameters, m, Id, dt, dx ,dy)
            
        np.save(local_dir + '/n_e_final_log10(eps) =%i'%np.floor(np.log10(eps))+ '_T =%i'%Tmax + '_dt =%4.3f'%dt, n)
        
        ######### Graphical outputs
        #graphical_output_3d_2d(eps*np.log(n*eps), X, Y, Tmax, '$x$', '$y$', local_dir, r'Final state - $u_\varepsilon$', 'u_e_log10(eps) =%i'%np.floor(np.log10(eps)))    
        #graphical_output_3d_2d(n, X, Y, Tmax, '$x$', '$y$', local_dir, r'Final state - $n_\varepsilon$', 'n_e_log10(eps) =%i'%np.floor(np.log10(eps)))
        graphical_output_contour_2d(T, Y, Rho_X, '$y$', '$t$',  local_dir, r'Dynamic - $\rho_\varepsilon^X(t, y)$', 'rhoX_e_log10(eps) =%i'%np.floor(np.log10(eps)))
        graphical_output_contour_2d(T, X, Rho_Y, '$x$', '$t$', local_dir, r'Dynamic - $\rho_\varepsilon^Y(t, x)$', 'rhoY_e_log10(eps) =%i'%np.floor(np.log10(eps)))
        #graphical_output_3d_2d(eps*np.log(np.transpose(Rho_X)), T, Y, Tmax, '$t$', '$y$', local_dir, r'Dynamic - $u^X_\varepsilon(t, y)$', 'uX_e_log10(eps) =%i'%np.floor(np.log10(eps)))
        #graphical_output_3d_2d(eps*np.log(np.transpose(Rho_Y)), T, X, Tmax, '$t$', '$x$', local_dir, r'Dynamic - $u^Y_\varepsilon(t, x)$', 'uY_e_log10(eps) =%i'%np.floor(np.log10(eps)))
        #graphical_output_2d(T, Rho, '$t$', r'$\rho_\varepsilon$', local_dir, r'Dynamic - $\rho_\varepsilon(t)$', 'rho_e_log10(eps) =%i'%np.floor(np.log10(eps)))        
        return()
    
    
#### Third function to update the numerical allelic distribution n according to a semi-explicit scheme
#### Called by the preivous one (run_selection) at each time step

def update(n, rho, rho_X, rho_Y, parameters, m , Id, dt, dx, dy):
    eps, Tmax, Nt, Nx, Ny, i0, j0, i1, j1, kappa, r, g = parameters
    
    rho_XY = mat_to_vect(np.diag(rho_Y)@np.ones((Nx, Ny))@np.diag(rho_X)/rho, Nx, Ny)
    nvect = mat_to_vect(n, Nx, Ny)
    nvect = scisplin.spsolve(Id + dt/eps*sp.spdiags(m-r/2 + kappa*rho*np.ones(Nx*Ny), 0, Nx*Ny, Nx*Ny),(nvect + dt/eps*r/2*rho_XY))
    n = vect_to_mat(nvect, Nx, Ny)
    rho_X = np.sum(n, axis = 0)*dx
    rho_Y = np.sum(n, axis = 1)*dy
    rho = np.sum(n)*dx*dy
    return(n, rho, rho_X, rho_Y)


#### Auxiliary functions

def Gauss(m, s, z):
    Nx=np.size(z)
    G = np.zeros(Nx)
    for k in range(Nx):
        G[k] = 1/( np.sqrt(2*np.pi)*s )* np.exp( - (z[k]-m)**2 / (2*(s)**2) )
    return(G)
    
### Creating directorydef create_directory(workdir,subworkdir,slimstordir):
def create_directory(workdir):
    try:
        # Create target Directory
        os.mkdir(workdir)
        print("Directory " , workdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , workdir ,  " already exists")
    
    return

#### Transform a vector into matrix of size (Nx, Ny)
def vect_to_mat(n, Nx, Ny):
    n_mat = np.zeros( (Nx, Ny) )
    for l in range(Ny):
        n_mat[:, l] = n[l*Nx:(l+1)*Nx]
    return(n_mat)        

#### Transform a matrix of size (Nx, Ny) into a vector
def mat_to_vect(n_mat, Nx, Ny):
    n = np.zeros( Nx*Ny )
    for l in range(Ny):
        n[l*Nx:(l+1)*Nx] = n_mat[:,l]
    return(n.flatten())   
    
    
### return a product of two normal densities with mean x0 and y0 and var sigmax^2 et sigmay^2
def gaussian_2d(X, Y, x0, y0, sigmax, sigmay):
    Nx = np.size(X)
    Ny = np.size(Y)
    gaussian = np.zeros((Nx, Ny))
    for j in range(Ny):
        gaussian[:,j] = 1/( 2*np.pi*sigmax*sigmay ) * np.exp(- (X - x0)**2/(2*sigmax**2)) * np.exp(- (Y[j] - y0)**2/(2*sigmay**2))
    return(gaussian)


### Selection functions
def sum_squared(x,y):
    return((x + y)**2)
def sum_of_squares(x,y):
    return(x**2 + y**2)
def one_minus_xy(x,y):
    return((1 - x*y)**2)

#### Various graphical output functions

#### 3D graphical output of n_mat at time Tfinal, saved in dir_path under filename
def graphical_output_3d_2d(n_mat, x, y, Tfinal, xlabel, ylabel, dir_path, title, filename):
    Ny = len(y)
    
    #### font and colors
    viridis = cm.get_cmap('viridis', Ny)
    ml.rcParams['mathtext.fontset'] = 'stix'
    ml.rcParams['font.family'] = 'STIXGeneral'
    plt.viridis()
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    for j in range(Ny):
        ax.plot(x, n_mat[:,j], zs=y[j], zdir = 'x', color = viridis( j ) )
        plt.tick_params(labelsize=25)
        plt.xlabel(ylabel, fontsize = 50, labelpad=25)
        plt.ylabel(xlabel, fontsize = 50, labelpad=25)
    #plt.title(title, fontsize = 20)
    plt.savefig(dir_path+"/"+filename+"_T = %4.1f"%Tfinal+".png")
    plt.show()
    plt.close()

#### 2D graphical output of Y_eps for times in T, saved in dir_path under filename
def graphical_output_2d_eps(T, Y, xlabel, ylabel, dir_path, title, eps, lneps, filename):
    
    #### font and colors
    Ny = len(Y)
    viridis = cm.get_cmap('viridis', Ny)
    ml.rcParams['mathtext.fontset'] = 'stix'
    ml.rcParams['font.family'] = 'STIXGeneral'
    plt.viridis()
    plt.figure(figsize=(10,10))
    plt.plot(T, Y, color = viridis(lneps*13) )
    plt.xlabel(ylabel, fontsize = 20)
    plt.ylabel(xlabel, fontsize = 20)
    plt.title(title+', $\log_{10}(e) = $ %4.1f'%np.log10(eps), fontsize = 20)
    plt.savefig(dir_path+"/"+filename+"_logeps = %4.1f"%np.log10(eps)+".png")
    plt.show()
    plt.close()

#### 2D graphical output of Y for times in T, saved in dir_path under filename
def graphical_output_2d(T, Y, xlabel, ylabel, dir_path, title, filename):
    
    #### font and colors
    Ny = len(Y)
    viridis = cm.get_cmap('viridis', Ny)    
    ml.rcParams['mathtext.fontset'] = 'stix'
    ml.rcParams['font.family'] = 'STIXGeneral'
    plt.figure(figsize=(5,5))
    plt.viridis()

    plt.plot(T, Y, color = viridis(10) )
    plt.xlabel(ylabel, fontsize = 20)
    plt.ylabel(xlabel, fontsize = 20)
    plt.title(title, fontsize = 20)
    plt.savefig(dir_path+"/"+filename+".png")
    plt.show()
    plt.close()
    
#### 2D graphical output of Y compared Ynum (same size) for times in T, saved in dir_path under filename
def graphical_output_2d_comp_eps(T, Y, Ynum, ylegend, ycomplegend, xlabel, ylabel, dir_path, title, filename, Eps):
    
    #### font and colors
    Ny = len(Y)
    viridis = cm.get_cmap('viridis', Ny)
    ml.rcParams['mathtext.fontset'] = 'stix'
    ml.rcParams['font.family'] = 'STIXGeneral'
    plt.figure(figsize=(8,8))
    plt.viridis()
    plt.plot(T, Y, color = viridis(10), label = ylegend)
    Neps = np.size(Eps)
    for i in range(Neps):
        plt.plot(T, Ynum[i,:], color = viridis(30*(i+1)), label = ycomplegend+r'_$\varepsilon = %4.2f'%Eps[i]+'$')
    plt.xlabel(xlabel, fontsize = 30)
    plt.ylabel(ylabel, fontsize = 30)
    plt.title(title, fontsize = 30)
    plt.legend(fontsize = 30)
    plt.savefig(dir_path+"/"+filename+".png")
    plt.show()
    plt.close()

def graphical_output_contour_2d(X, Y, M, xlabel, ylabel, dir_path, title, filename):
    #### font and colors   
    ml.rcParams['mathtext.fontset'] = 'stix'
    ml.rcParams['font.family'] = 'STIXGeneral'
    plt.figure(figsize=(8,8))
    plt.viridis()
    plt.contourf(X, Y, M, 100)
    plt.xlabel(ylabel, fontsize = 40)
    plt.ylabel(xlabel, fontsize = 40)
    plt.title(title, fontsize = 40)
    plt.tick_params(labelsize=30)

    plt.savefig(dir_path+"/"+filename+".png")
    plt.show()
    plt.close()