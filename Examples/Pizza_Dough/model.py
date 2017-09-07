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

#
# EXAMPLE PDE:
#
#initial = lambda x: 100.0*x*(1.0-x)
#source = lambda x,t: 100.0*((np.power(x,2) - x)/(np.power(t+1,2)) + 2.0/(t+1))
#true_solution = lambda x,t: 100.0*(x - np.power(x,2))/(t+1)
#


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
    center = np.power(np.sin( 2.0*np.pi*(c0 + rate*t)),2)
    radius = r0
    a = center - radius
    b = center + radius
    scaling = amp/np.exp(1.0/(-np.power(radius,2)))
    if (x<=a+e) or (x>=b-e):
        val = 0.0
    else:
        val = scaling*np.exp(1.0/(np.power(x-center,2) - np.power(radius,2)))
    return val


# Define true solution if known
#initial = lambda x: 100.0*x*(1.0-x)

#initial1 = np.vectorize(lambda x: impulse(x,0.0,0.4,amp=15.0))
#initial2 = np.vectorize(lambda x: impulse(x, 0.4,0.6,amp=5.0))
#initial3 = np.vectorize(lambda x: impulse(x, 0.6,1.0,amp=25.0))
#initial = lambda x: initial1(x) + initial2(x) + initial3(x)

initial = lambda x: 10.0*(np.power(x,0.10) * np.power(1-x,0.10) )

# Define source term for differential equation
#source = lambda x,t: x*(1.0-x)/(1.0+t)
#source = lambda x,t: 100.0*((np.power(x,2) - x)/(np.power(t+1,2)) + 2.0/(t+1))
#source = lambda x,t: -10.0

#rate = 2.0
#source_tmp = np.vectorize(lambda x: impulse(x,0.0,0.4,amp=100.0))
#source = np.vectorize(lambda x,t: source_tmp(x-t*rate))
source = np.vectorize(lambda x,t: moving_impulse(x,t,0.4,0.4,rate=2.0,amp=(1000.0*(1.0-t*3.0/4.0))))

# Define true solution if known
#true_solution = lambda x,t: 100.0*(x - np.power(x,2))/(t+1)



# Define mesh parameters for finite element space
mesh_start = 0.0
mesh_end = 1.0
mesh_size = 21
mesh = np.linspace(mesh_start, mesh_end, mesh_size)

t_start = 0.0
t_end = 1.25
t_steps = 500
times = np.linspace(t_start, t_end, t_steps+1)


# Check stability condition
dx = mesh[1] - mesh[0]
dt = times[1] - times[0]

if dt >= 0.5*np.power(dx,2):
    print('\n\n')
    print('-------------------------------------------------')
    print(' WARNING: stability condition is not satisfied')
    print('       dt = %0.5f       (dx)^2 = %0.5f'  %(dt,np.power(dx,2)) )
    print('-------------------------------------------------')


# Define mesh parameters for evaluation
eval_mesh_size = 100
eval_mesh = np.linspace(mesh_start, mesh_end, eval_mesh_size)


# Verify initial conditions are correctly specified
#tmp_init_vals = initial(eval_mesh)
#plt.plot(eval_mesh,tmp_init_vals,'r')
#plt.show()



#
# Solve PDE using Finite Element Method
#

# Initialize finite element space
space = fem_space(mesh, times, basis, initial, source)

# Construct stiffness matrix
space.form_stiffness()

# Construct mass matrix
space.form_mass()

# Compute Cholesky factorization of mass matrix
space.compute_chol_mass()

# Compute Cholesky factorization for time stepping
space.compute_chol()

# Solve system for coefficients in FEM expansion
soln_coeffs = space.solve_system()

# Evaluate solution
soln_vals = space.evaluate_mesh(eval_mesh, times, soln_coeffs)


# Evaluate source term
mesh_time = np.meshgrid(eval_mesh,times)
source_vals = source(mesh_time[0], mesh_time[1])
source_vals = np.transpose(source_vals)

# Evaluate solution (if available)
try:
    true_solution
except:
    true_vals = None
else:
    true_vals = true_solution(eval_mesh[:,None], times[None,:])
    #print(true_vals)


# Verify initial conditions are correctly interpolated
#plt.plot(eval_mesh,soln_vals[:,0],'b')
#plt.plot(eval_mesh,true_vals[:,0],'r--')
#plt.show()

















# Plot prediction (and true solution if available)
#print('\n  [ Creating Animation ]\n')
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



def init():
    line1.set_ydata(np.ma.array(eval_mesh, mask=True))
    line2.set_ydata(np.ma.array(eval_mesh, mask=True))
    return line1, line2, title,

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
