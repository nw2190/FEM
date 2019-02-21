from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Load 'finite_elements.py' to define finite element space
from finite_elements import *


# Specify basis functions
#basis = linear_basis
basis = quadratic_basis


# Define source term for differential equation
#source = lambda x: 300.0*(2.0*x - 1.0)


# Define true solution if known
#true_solution = lambda x: 100*(x-0.0)*(x-1.0)*(x-0.5)

# EXAMPLE ODE 1:  u(x) = x*(x-1.0)*sin(x)*cos(x)
#source = lambda x: -2*(x-1)*np.power(np.sin(x),2) - 2*x*np.power(np.sin(x),2) + 2*(x-1)*np.power(np.cos(x),2) + 2*x*np.power(np.cos(x),2) - 4*(x-1)*x*np.sin(x)*np.cos(x) + 2*np.sin(x)*np.cos(x)
#true_solution = lambda x: x*(x-1.0)*np.sin(x)*np.cos(x)

# EXAMPLE ODE 2:  u(x) = A*sin(2*pi*freq*x)
freq = 5.0
amp = 10.0
source = lambda x: -amp*np.power(2*np.pi*freq,2)*np.sin(2*np.pi*freq*x)
true_solution = lambda x: amp*np.sin(2*np.pi*freq*x)



# Define mesh parameters for finite element space
mesh_start = 0.0
mesh_end = 1.0
mesh_size = 51
mesh = np.linspace(mesh_start, mesh_end, mesh_size)

# Define mesh parameters for evaluation
eval_mesh_size = 200
eval_mesh = np.linspace(mesh_start, mesh_end, eval_mesh_size)




# Initialize finite element space
space = fem_space(mesh, basis)

# Construct stiffness matrix
space.form_stiffness()

# Compute Cholesky factorization of stiffness matrix
space.compute_chol()

# Construct right-hand-side of weak formulation
space.form_rhs(source)

# Solve system for coefficients in FEM expansion
soln_coeffs = space.solve_system()

# Evaluate solution and source term on eval_mesh
soln_vals = space.evaluate_mesh(eval_mesh, soln_coeffs)
source_vals = source(eval_mesh)


# Plot source term
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(eval_mesh, source_vals, 'r')
source_title = 'Source Term'
ax1.set_title(source_title)
plt.xlabel('x - axis')
plt.ylabel('y - axis')


# Plot prediction (and solution if available)
try:
    true_solution
except:
    ax2 = fig.add_subplot(122)
    ax2.plot(eval_mesh, soln_vals)
    soln_title = 'Approximate Solution'
    ax2.set_title(soln_title)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
      
else:
    ax2 = fig.add_subplot(122)
    true_vals = true_solution(eval_mesh)
    ax2.plot(eval_mesh, soln_vals, 'b')
    ax2.plot(eval_mesh, true_vals, 'r--')
    soln_title = 'Approximate Solution'
    ax2.set_title(soln_title)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')


plt.show()




PLOT_BASIS = False

if PLOT_BASIS:
    # Plot basis functions
    plot_count = 3
    for n in range(0,plot_count):
        basis_vals = space.evaluate_basis_mesh(eval_mesh, n)
        plt.plot(eval_mesh, basis_vals)
        plt.show()

    # Plot basis gradient functions
    plot_count = 3
    for n in range(0,plot_count):
        basis_grad_vals = space.evaluate_basis_grad_mesh(eval_mesh, n)
        plt.plot(eval_mesh, basis_grad_vals)
        plt.show()

