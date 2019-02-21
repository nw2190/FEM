from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Load 'finite_elements.py' to define finite element space
from finite_elements import *

# Load 'basis_functions.py' to define basis functions
from basis_functions import *


# Specify basis functions
basis = linear_basis
#basis = quadratic_basis

# Specify time-stepping scheme
#time_stepping = 'Backward Euler'
time_stepping = 'Crank-Nicolson'

#Specify solver method
#solve_method='Cholesky'
solve_method='Conjugate Gradient'

# Define mesh for finite element space
mesh_start = 0.0
mesh_end = 1.0
mesh_size = 101
mesh = np.linspace(mesh_start, mesh_end, mesh_size)

# Define mesh for time stepping
t_start = 0.0
t_end = 0.15
t_steps = 101
times = np.linspace(t_start, t_end, t_steps+1)

# Define mesh for solution evaluation
eval_mesh_size = 150
eval_mesh = np.linspace(mesh_start, mesh_end, eval_mesh_size)



###
# EXAMPLE PDE I:
###
#initial = lambda x: 100.0*x*(1.0-x)
#source = lambda x,t: 100.0*((np.power(x,2) - x)/(np.power(t+1,2)) + 2.0/(t+1))
#true_solution = lambda x,t: 100.0*(x - np.power(x,2))/(t+1)


###
# EXAMPLE PDE II:
###
#initial = lambda x: np.sin(2.0*np.pi*x)*np.sin(2.0*np.pi*(0.0+x))
#source = lambda x,t: 2.0*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*(x+t)) - 8.0*np.power(np.pi,2)*np.cos(2*np.pi*(t+2.0*x))
#true_solution = lambda x,t: np.sin(2*np.pi*x)*np.sin(2.0*np.pi*(x+t))


###
# EXAMPLE PDE III:
###
#initial = lambda x: x*(1-x)*np.sin(8*np.pi*(x-0.0))
#source = lambda x,t: 8*np.pi*(x-1)*x*np.cos(8*np.pi*(x - t))  - (-64*np.power(np.pi,2)*(1-x)*x*np.sin(8*np.pi*(x-t)) - 2*np.sin(8*np.pi* (x-t)) + 16*np.pi*(1- 2*x)*np.cos(8*np.pi*(x - t)))
#true_solution = lambda x,t: x*(1-x)*np.sin(8*np.pi*(x-t))


###
# EXAMPLE PDE IV:
###
rate_factor = 7.0
initial = lambda x: x*(1-x)*np.sin(2*rate_factor*np.pi*(x-0.0))
source = lambda x,t: 2*rate_factor*np.pi*(x-1)*x*np.cos(2*rate_factor*np.pi*(x - t))  - (np.power(x,2)*(-np.power(2*rate_factor,2)*np.power(np.pi,2)*(1-x)*x*np.sin(2*rate_factor*np.pi*(x-t)) - 2*np.sin(2*rate_factor*np.pi*(x-t)) + 4*rate_factor*np.pi*(1-x)*np.cos(2*rate_factor*np.pi*(x-t)) - 4*rate_factor*np.pi*x*np.cos(2*rate_factor*np.pi*(x-t))) + 2*x*((1-x)*np.sin(2*rate_factor*np.pi*(x-t)) - x*np.sin(2*rate_factor*np.pi*(x-t)) + 2*rate_factor*np.pi*(1-x)*x*np.cos(2*rate_factor*np.pi*(x-t))))
stiffness = lambda x: np.power(x,2)
true_solution = lambda x,t: x*(1-x)*np.sin(2*rate_factor*np.pi*(x-t))



#
# Define PDE problem statement
#

# Define initial values
#initial = lambda x: 100.0*x*(1.0-x)

# Define source term for differential equation
#source = lambda x,t: x*(1.0-x)/(1.0+t)

# Define stiffness term for differential equation (if applicable)
#stiffness = None

# Define true solution if known
#true_solution = lambda x,t: 100.0*(x - np.power(x,2))/(t+1)



#
# Check stability condition
#

#dx = mesh[1] - mesh[0]
#dt = times[1] - times[0]

#if dt >= 0.5*np.power(dx,2):
#    print('\n -----------------------------------------------')
#    print('| WARNING: stability condition is not satisfied |')
#    print('|     dt = %0.5f       (dx)^2 = %0.5f       |'  %(dt,np.power(dx,2)) )
#    print(' -----------------------------------------------\n')



# Verify initial conditions are correctly specified
#tmp_init_vals = initial(eval_mesh)
#plt.plot(eval_mesh,tmp_init_vals,'r')
#plt.show()



#
# Solve PDE using Finite Element Method
#

# Initialize finite element space
space = fem_space(mesh, times, basis, initial, source, stiffness=stiffness, time_stepping=time_stepping, solve_method=solve_method)

# Construct stiffness matrix
space.form_stiffness()

# Construct mass matrix
space.form_mass()

# Compute Cholesky factorization of mass matrix
space.compute_chol_mass()
    
# Solve system for coefficients in FEM expansion
soln_coeffs = space.solve_system()

# Evaluate solution
soln_vals = space.evaluate_mesh(eval_mesh, times, soln_coeffs)


# Evaluate source term
#mesh_time = np.meshgrid(eval_mesh,times)
#source_vals = source(mesh_time[0], mesh_time[1])
#source_vals = np.transpose(source_vals)

# Evaluate solution (if available)
try:
    true_solution
except:
    true_vals = None
else:
    true_vals = true_solution(eval_mesh[:,None], times[None,:])


# Verify initial conditions are correctly interpolated
#plt.plot(eval_mesh,soln_vals[:,0],'b')
#plt.plot(eval_mesh,true_vals[:,0],'r--')
#plt.show()



# Plot prediction (and true solution if available)
fig, ax = plt.subplots()
line1, = ax.plot(eval_mesh, soln_vals[:,0])

try:
    true_solution
except:
    line2, = ax.plot(eval_mesh, soln_vals[:,0],'b')
else:
    line2, = ax.plot(eval_mesh, true_vals[:,0],'r--')
    
ax.set_ylim([np.min(soln_vals),np.max(soln_vals)])


# Define title/label
soln_title = 't = %0.2f' %(times[0])

from_left = 0.85
from_bottom = 0.875
title = ax.text(from_left,from_bottom, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                fontsize=20, transform=ax.transAxes, ha="center")
plt.xlabel('x - axis', fontsize=20)
plt.ylabel('y - axis', fontsize=20)


# Define initialization fram for animation
def init():
    line1.set_ydata(np.ma.array(eval_mesh, mask=True))
    line2.set_ydata(np.ma.array(eval_mesh, mask=True))
    return line1, line2, title,

# Define animation step
def animate(i):
    line1.set_ydata(soln_vals[:,i])
    try:
        true_solution
    except:
        line2.set_ydata(soln_vals[:,i])
    else:
        line2.set_ydata(true_vals[:,i])
    soln_title = 't = %0.2f' %(times[i])
    title.set_text(soln_title)

    return line1, line2, title,

animation_delay = int(np.max([np.floor(-190.0/990.0*(t_steps-10.0)) + 200.0, 20.0]))
ani = animation.FuncAnimation(fig, animate, np.arange(1, t_steps), init_func=init,
                              interval=animation_delay, blit=True)

plt.show()

print('  [ Saving Animation ]\n\n')
filename = 'animation.mp4'
ani.save(filename, writer=None, fps=None, dpi=None, codec=None, bitrate=None)
















###
#  Miscellaneous code for problem setup
##




def impulse(x,a,b,amp=1.0):
    e = 0.000001
    center = (a+b)/2.0
    radius = (b-a)/2.0
    scaling = amp/np.exp(1.0/(-np.power(radius,2)))
    if (x<=a+e) or (x>=b-e):
        val = 0.0
    else:
        val = scaling*np.exp(1.0/(np.power(x-center,2) - np.power(radius,2)))
    return val

def moving_impulse(x,t,c0,r0,rate=1.0,amp=1.0):
    e = 0.000001
    #center = np.power(np.sin( 2.0*np.pi*(c0 + rate*t)),2)
    center = np.power(np.sin( 2.0*np.pi*(rate*t)),2)
    radius = r0
    a = center - radius
    b = center + radius
    scaling = amp/np.exp(1.0/(-np.power(radius,2)))
    if (x<=a+e) or (x>=b-e):
        val = 0.0
    else:
        val = scaling*np.exp(1.0/(np.power(x-center,2) - np.power(radius,2)))
    return val



# Define initial values
#initial = lambda x: 100.0*x*(1.0-x)

#initial1 = np.vectorize(lambda x: impulse(x,0.0,0.4,amp=15.0))
#initial2 = np.vectorize(lambda x: impulse(x, 0.4,0.6,amp=5.0))
#initial3 = np.vectorize(lambda x: impulse(x, 0.6,1.0,amp=25.0))
#initial = lambda x: initial1(x) + initial2(x) + initial3(x)
#initial = lambda x: 100.0*(initial1(x) + initial2(x) + initial3(x))

#initial = lambda x: 10.0*(np.power(x,0.10) * np.power(1-x,0.10) )

# Define source term for differential equation
#source = lambda x,t: x*(1.0-x)/(1.0+t)
#source = lambda x,t: 100.0*((np.power(x,2) - x)/(np.power(t+1,2)) + 2.0/(t+1))
#source = lambda x,t: -10.0

#rate = 2.0
#source_tmp = np.vectorize(lambda x: impulse(x,0.0,0.4,amp=100.0))
#source = np.vectorize(lambda x,t: source_tmp(x-t*rate))
#source = np.vectorize(lambda x,t: moving_impulse(x,t,0.4,0.4,rate=0.75,amp=(1000.0*(1.0-t*3.0/4.0))))



# Define stiffness term for differential equation (if applicable)
#stiffness = None
#stiffness = np.vectorize(lambda x: 2.0 + 2.0*x*(x-1.0))

# Define true solution if known
#true_solution = lambda x,t: 100.0*(x - np.power(x,2))/(t+1)


