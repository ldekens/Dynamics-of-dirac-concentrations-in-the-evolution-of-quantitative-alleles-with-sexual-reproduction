#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 01:49:15 2021

@author: dekens
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 09:40:15 2020

@author: dekens
"""

import numpy as np
import os
import matplotlib as ml
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import sparse as sp
import scipy.sparse.linalg as scisplin
from multiprocessing import Pool
from itertools import repeat

### First function called in main code, run in parallel for selection in Selection

def run_theoretical_canonical_equation_comparison(Selection, parameters, path, X_0, Y_0, N_run):
    p  = Pool(3)
    inputs = [*zip(Selection, repeat(parameters), repeat(path), repeat(X_0), repeat(Y_0), repeat(N_run))]
    p.starmap(run_theoretical_canonical_equation_comparison_selection, inputs)

### Function called by the previous one, per selection function (selection). 
### Create the graphical output for the given selection function.
### To do so, run the comparison between numerical and analytical trajectories of the dominant alleles given an initial distribution, through run_per_IC_comparison. Next plot the trajectories

def run_theoretical_canonical_equation_comparison_selection(selection, parameters, path, X_0, Y_0, N_run):
    if selection == 'sum_of_squares':
        m, dx_m, dy_m, dxx_m, dyy_m = sum_of_squares, dx_sum_of_squares, dy_sum_of_squares, dxx_sum_of_squares, dyy_sum_of_squares
        addplot = False
    if selection == 'square_of_sum':
        m, dx_m, dy_m, dxx_m, dyy_m = square_of_sum, dx_square_of_sum, dy_square_of_sum, dxx_square_of_sum, dyy_square_of_sum
        addplot = True
        X_addplot = np.linspace(-2, 2, num = 1000)
        Y_addplot = -X_addplot
    if selection == 'one_minus_xy':
        m, dx_m, dy_m, dxx_m, dyy_m = one_minus_xy, dx_one_minus_xy, dy_one_minus_xy, dxx_one_minus_xy, dyy_one_minus_xy
        addplot = True
        X_addplot = np.linspace(-2, 2, num = 1000)
        Y_addplot = 1/X_addplot
    
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\bar{x}$', fontsize=50)
    ax.set_ylabel(r'$\bar{y}$', fontsize=50)
    ax.set_ylim((-2, 2))
    ax.set_xlim((-2, 2))
    ax.tick_params(axis="y", labelsize=40)
    ax.tick_params(axis="x", labelsize=40)
    ml.rcParams['mathtext.fontset'] = 'stix'
    ml.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams.update({"text.usetex": True})
    viridis = cm.get_cmap('viridis', N_run + 2)
    
    # Loop over the N_run different initial states randomly drawn in the main code
    for n in range(N_run):
        x, y, xnum, ynum = run_per_IC_comparison(X_0[n], Y_0[n], m, dx_m, dy_m, dxx_m, dyy_m, parameters)
        n_color = np.random.randint(1, N_run + 1)
        ax.plot(x, y, color = viridis(n_color), linewidth = 5, linestyle = 'dashed')
        ax.plot(xnum, ynum, color = viridis(n_color), linewidth = 5)
        ax.scatter(x[-1], y[-1], color = viridis(n_color), marker = 'x', s = 400)
        ax.scatter(xnum[-1], ynum[-1], color = viridis(n_color), s = 400)
        if addplot:
            ax.plot(X_addplot[X_addplot > 0], Y_addplot[X_addplot > 0], color = 'black', linewidth = 1)
            ax.plot(X_addplot[X_addplot < 0], Y_addplot[X_addplot < 0], color = 'black', linewidth = 1)
    plt.show()
    plt.savefig(path +'/'+ selection +'.png')
    plt.close()

### Third function, called by the previous one  (run_theoretical_canonical_equation_comparison_selection)
### Return four vectors x, y, xnum, ynum. (x, y)  is the temporal trajectories of the dominant alleles according to the analytical predictions, (xnum, ynum) are the ones according to the numerical resolution of the equation.
### Calls the function run_iteration_comparison, which computes the the next step of the trajectories given a time point.

def run_per_IC_comparison(x0, y0, m, dx_m, dy_m, dxx_m, dyy_m, parameters):
    r, g, kappa, eps = parameters
    # Time
    Tmax = 20
    Nt = np.floor(Tmax*10/eps).astype(int)
    T, dt = np.linspace(0, Tmax, Nt, retstep = True)
    
    # Allelic space
    Nx = 201
    Ny = 201
    X, dx = np.linspace(-2, 2, Nx, retstep = True)
    Y, dy = np.linspace(-2, 2, Ny, retstep = True)
    
    # Loop over the trait space to build the matrix that holds the selection values
    M = np.zeros((Nx,Ny))
    for j in range(Ny):
        for i in range(Nx):
            M[i, j] = g*m(X[i], Y[j])
    m = mat_to_vect(M, Nx, Ny)
    
    x, y, xnum, ynum = np.zeros(Nt), np.zeros(Nt), np.zeros(Nt), np.zeros(Nt)
    n = gaussian_2d(X, Y, x0, y0, np.sqrt(eps), np.sqrt(eps))
    x[0], y[0], xnum[0], ynum[0], dxx_uY, dyy_uX = x0, y0, x0, y0, -2, -2
    
    # Loop avec time grid
    for nt in range(Nt-1):
        x[nt+1], y[nt+1], xnum[nt+1], ynum[nt+1], n, dxx_uY, dyy_uX = run_iteration_comparison(x[nt], y[nt], n, dxx_uY, dyy_uX, m, dx_m, dy_m, dxx_m, dyy_m, dt, X, dx, Y, dy, parameters)
    return(x, y, xnum, ynum)

### Fourth function, called by the previous one (run_per_IC_comparison)
### Compute the next values of x, y, xnum, ynum, given the ones at the previous time step, and the allelic distribution n at the previous time step

def run_iteration_comparison(x, y, n, dxx_uY, dyy_uX, m, dx_m, dy_m, dxx_m, dyy_m, dt, X, dx, Y, dy, parameters):
    
    ## Numerical analysis of canonical ODE
    dxx_uY_next = dxx_uY - dt*dxx_m(x, y) 
    dyy_uX_next = dyy_uX - dt*dyy_m(x, y)
    x_next = x + dt*dx_m(x, y)/ (dxx_uY)
    y_next = y + dt*dy_m(x, y)/ (dyy_uX)
    
    ## Numerical analysis of n
    
    r, g, kappa, eps = parameters
    Nx, Ny = np.size(X), np.size(Y)
    
    # Auxiliary quantities
    rho_X = np.sum(n, axis = 0)*dx
    rho_Y = np.sum(n, axis = 1)*dy
    rho = np.sum(n)*dx*dy
    Id = sp.spdiags(np.ones(Nx*Ny), 0, Nx*Ny, Nx*Ny)
    rho_XY = mat_to_vect(np.diag(rho_Y)@np.ones((Nx,Ny))@np.diag(rho_X)/rho, Nx, Ny)
    
    # Compute new allelic distribution n according the resolution of the equation
    nvect = mat_to_vect(n, Nx, Ny)
    nvect = scisplin.spsolve(Id+dt/eps*sp.spdiags(m - r/2 + kappa*rho*np.ones(Nx*Ny), 0, Nx*Ny, Nx*Ny),(nvect + dt/eps*r/2*rho_XY))
    n = vect_to_mat(nvect, Nx, Ny)
    
    # Compute the dominant alleles as maxximizing the marginal distributions rho_X and rho_Y
    rho_X = np.sum(n, axis = 0)*dx
    rho_Y = np.sum(n, axis = 1)*dy
    xnum = X[np.argmax(rho_Y)]
    ynum = Y[np.argmax(rho_X)]
    return(x_next, y_next, xnum, ynum, n, dxx_uY_next, dyy_uX_next)

######### Auxiliary functions

### Return Gaussian vector with grid z, mean m and standard deviation s
def Gauss(m, s, z):
    Nx=np.size(z)
    G = np.zeros(Nx)
    for k in range(Nx):
        G[k] = 1/( np.sqrt(2*np.pi)*s )* np.exp( - (z[k]-m)**2 / (2*(s)**2) )
    return(G)
    
### Creating directory
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

### Some analytical selection functions
def sum_of_squares(x,y):return((x**2+y**2))
def dx_sum_of_squares(x,y):return(2*x)
def dy_sum_of_squares(x,y):return(2*y)
def dxx_sum_of_squares(x,y):return(2)
def dyy_sum_of_squares(x,y):return(2)
    

def square_of_sum(x,y):return((x+y)**2)
def dx_square_of_sum(x,y):return(2*(x+y))
def dy_square_of_sum(x,y):return(2*(x+y))
def dxx_square_of_sum(x,y):return(2)
def dyy_square_of_sum(x,y):return(2)

def one_minus_xy(x,y):return((1-x*y)**2)
def dx_one_minus_xy(x,y):return(-y*2*(1-x*y))
def dy_one_minus_xy(x,y):return(-x*2*(1-x*y))
def dxx_one_minus_xy(x,y):return(y**2*2)
def dyy_one_minus_xy(x,y):return(x**2*2)





 