from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import time

# Load 'finite_elements.py' to define finite element space
from finite_elements import *


# Specify basis functions
#basis = linear_basis
basis = quadratic_basis



def impulse(x,a,b,amp=1.0):
    e = 0.000001
    center = (a+b)/2.0
    radius = (b-a)/2.0
    scaling = amp/np.exp(1.0/(-np.power(radius,2)))

    if x<=a+e:
        val = 0.0
    elif x>=b-e:
        val = 0.0
    else:
        val = scaling*np.exp(1.0/(np.power(x-center,2) - np.power(radius,2)))
    return val


# Define true solution if known
#initial = lambda x: 100.0*x*(1.0-x)

initial1 = np.vectorize(lambda x: impulse(x,-0.1,0.6,amp=2.0))
initial2 = np.vectorize(lambda x: impulse(x, 0.4,1.1,amp=5.0))
initial = lambda x: initial1(x)+initial2(x)

# Define true solution if known
#initial = lambda x: 100.0*x*(1.0-x)
#initial = lambda x: 100.0*x*(1.0-x)

# Define source term for differential equation
source = lambda x,t: 100.0*((np.power(x,2) - x)/(np.power(t+1,2)) - 2.0/(t+1))

# Define mesh parameters for finite element space
mesh_start = 0.0
mesh_end = 1.0
mesh_size = 21
mesh = np.linspace(mesh_start, mesh_end, mesh_size)

t_start = 0.0
t_end = 0.10
t_steps = 11
times = np.linspace(t_start, t_end, t_steps+1)


# Define mesh parameters for evaluation
eval_mesh_size = 100
eval_mesh = np.linspace(mesh_start, mesh_end, eval_mesh_size)



# Initialize finite element space
space = fem_space(mesh, times, basis, initial, source)

# Construct mass matrix
space.form_mass()

# Compute Cholesky factorization of mass matrix
space.compute_chol_mass()

# Compute coefficients for FEM projection
init_coeffs = space.proj_func(initial)

# Evaluate solution and source term on eval_mesh
M = eval_mesh.size
init_vals = np.zeros([M])

for m in range(0,M):
    pt = eval_mesh[m]
    init_vals[m] = space.evaluate_point(pt, init_coeffs)

true_vals = initial(eval_mesh)


plt.plot(eval_mesh,init_vals,'b')
plt.plot(eval_mesh,true_vals,'r--')
plt.show()

    


