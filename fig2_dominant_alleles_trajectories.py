#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 15:22:15 2021

@author: dekens
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import fig2_tools as tools

######### Title

title = "fig2_dominant_alleles_trajectories"
workdir = title
tools.create_directory(workdir)

Mfun = [tools.sum_of_squares, tools.sum_squared, tools.one_minus_xy]
Mfun_string  = ['x^2+y^2', '(x+y)^2', '(1-xy)^2']
Iinf, Isup = -2, 2
parameters = 0.05, 100, 10000, 401, 401, 170, 330, 270, -250, 1, 1, 1/8 ##  eps, Tmax, Nt, Nx, Ny, i0, j0, i1, j1, kappa, r, g

tools.run_Selection(Mfun, Mfun_string, Iinf, Isup, parameters, workdir)


    
