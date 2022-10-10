        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:42:40 2021

@author: dekens
"""
"""
Created on Wed Sep  8 17:13:06 2021

@author: dekens
"""
import numpy as np
import tools_fig3 as tools
#### Title

title = 'fig3__phase_space'
path = title
tools.create_directory(path)

#### Allelic space

r, g, kappa, eps = 1, 1, 1, 0.01
parameters = r, g, kappa, eps

Selection = ['one_minus_xy', 'square_of_sum', 'sum_of_squares']

#### Initial monomorphic states' coordinates: randomly uniformly drawn over the trait space [-2, 2]x[-2, 2]
N_run = 20
X_0 = np.random.uniform(-2, 2, size = N_run)
Y_0 = np.random.uniform(-2, 2, size = N_run)


tools.run_theoretical_canonical_equation_comparison(Selection, parameters, path, X_0, Y_0, N_run)
