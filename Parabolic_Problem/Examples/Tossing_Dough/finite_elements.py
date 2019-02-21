from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
import sys


class linear_basis:
    # Initialize the class
    def __init__(self):
        self.name = 'Linear Basis Functions'
        self.elmnt_types = 1
        self.support = 2
        self.support_overlap = 1

    def func(self,x,a,b,el_type):
        if (x<=a) or (x>=b):
            val = 0.0
        elif x<=(a+b)/2.0:
            val = 2.0/(b-a)*(x-a)
        else:
            val = 2.0/(b-a)*(b-x)
        return val
        
    def grad(self,x,a,b,el_type):
        if (x<=a) or (x>=b):
            val = 0.0
        elif x<=(a+b)/2.0:
            val = 2.0/(b-a)
        else:
            val = 2.0/(a-b)
        return val


class quadratic_basis:
    # Initialize the class
    def __init__(self):
        self.name = 'Quadratic Basis Functions'
        self.elmnt_types = 2
        self.support = 2        
        self.support_overlap = 3
        

    def func(self,x,a,b,el_type):
        if el_type == 0:
            if (x<=a) or (x>=b):
                val = 0.0
            else:
                scaling = (b-a)/2.0 * (a-b)/2.0
                val = (x-a)*(x-b)/scaling
            return val
        
        elif el_type == 1:
            if (x<=a) or (x>=b):
                val = 0.0
            elif x<=(a+b)/2.0:
                scaling = ((a+b)/2.0-a)*((a+b)/2.0-(3*a-b)/2.0)
                val = (x-a)*(x-(3*a-b)/2.0)/scaling
            else:
                scaling = ((a+b)/2.0-b)*((a+b)/2.0-(5*b-a)/2.0)
                val = (x-b)*(x-(5*b-a)/2.0)/scaling
            return val

    def grad(self,x,a,b,el_type):
        if el_type == 0:
            if (x<=a) or (x>=b):
                val = 0.0
            else:
                scaling = (b-a)/2.0 * (a-b)/2.0
                val = ((x-b) + (x-a))/scaling
            return val
        
        elif el_type == 1:
            if (x<=a) or (x>=b):
                val = 0.0
            elif x<=(a+b)/2.0:
                scaling = ((a+b)/2.0-a)*((a+b)/2.0-(3*a-b)/2.0)
                val = ((x-(3*a-b)/2.0) + (x-a))/scaling
            else:
                scaling = ((a+b)/2.0-b)*((a+b)/2.0-(5*b-a)/2.0)
                val = ((x-(5*b-a)/2.0) + (x-b))/scaling
            return val



class fem_space:
    # Initialize the class
    def __init__(self, mesh, times, basis_functions, initial, source):
        self.mesh = mesh
        self.N = mesh.size - 2
        self.a = mesh[0]
        self.b = mesh[-1]

        self.times = times
        self.T = times.size
        self.dt = times[1] - times[0]
        self.start_time = times[0]
        self.final_time = times[-1]
        self.current_step = 0

        self.basis = basis_functions()
        self.basis.funcs = self.init_funcs()
        self.basis.grads = self.init_grads()
        self.dim = self.basis.elmnt_types*self.N

        self.initial = initial
        self.source = source

        self.soln_coeffs = np.zeros([self.dim, self.T])
        
        print('\n Finite Element Space:  |')
        print('-------------------------\n')
        print('   Spatial Domain = [%0.2f, %0.2f]' %(self.a,self.b))
        print('   Time Interval = [%0.2f, %0.2f]' %(self.start_time,self.final_time))
        print('   Interior Nodes = %d' %(self.N))
        print('   FEM Basis = %s' %(self.basis.name))
        print('   Problem Dimension = %d\n' %(self.dim))

        print('\n FEM Solver Progress:   |')
        print('-------------------------')


    # Initialize basis functions
    def init_funcs(self):
        mesh = self.mesh
        N = self.N
        support = self.basis.support
        elmnt_types = self.basis.elmnt_types
        funcs = []
        for n in range(0,N):
            for el_type in range(0,elmnt_types):
                func = lambda x, n=n, el_type=el_type: self.basis.func(x,mesh[n],mesh[n+support],el_type)
                funcs.append(func)
        return funcs

    # Initialize gradients of basis functions
    def init_grads(self):
        mesh = self.mesh
        N = self.N
        support = self.basis.support
        elmnt_types = self.basis.elmnt_types
        grads = []
        for n in range(0,N):
            for el_type in range(0,elmnt_types):
                func = lambda x, n=n, el_type=el_type: self.basis.grad(x,mesh[n],mesh[n+support],el_type)
                grads.append(func)
        return grads

        
    # Form stiffness matrix
    def form_stiffness(self):
        print('\n   [ Forming Stiffness Matrix ]\n')
        mesh = self.mesh
        dim = self.dim
        a = self.a
        b = self.b
        grads = self.basis.grads
        support = self.basis.support
        support_overlap = self.basis.support_overlap
        elmnt_types = self.basis.elmnt_types
        
        A = np.zeros([dim,dim])
        for n in range(0,dim):
            
            if n < dim-support_overlap:
                overlap = support_overlap
            else:
                overlap = (dim-n) - 1
                
            for m in range(n,n+overlap+1):
                #print('[%d,%d]' %(n,m))
                lo_index = int(np.floor(n/elmnt_types))
                hi_index = np.min( [int(np.floor(m/elmnt_types) + 2.0*support), self.N+1])
                lo_lim = mesh[lo_index]
                hi_lim = np.min( [ mesh[hi_index], mesh[-1]])
                basis_grad_fn1 = grads[n]
                basis_grad_fn2 = grads[m]
                func = lambda x: basis_grad_fn1(x)*basis_grad_fn2(x)
                val = quad(func, lo_lim, hi_lim, limit=100)[0]
                A[n,m] = A[m,n] = val
            #sys.stdout.write('Row: {0} of {1}\r'.format(n+1,dim))

        self.A = A


    # Form mass matrix
    def form_mass(self):
        print('\n   [ Forming Mass Matrix ]\n')
        mesh = self.mesh
        dim = self.dim
        a = self.a
        b = self.b
        funcs = self.basis.funcs
        support = self.basis.support
        support_overlap = self.basis.support_overlap
        elmnt_types = self.basis.elmnt_types
        
        B = np.zeros([dim,dim])
        for n in range(0,dim):
            
            if n < dim-support_overlap:
                overlap = support_overlap
            else:
                overlap = (dim-n) - 1
                
            for m in range(n,n+overlap+1):
                lo_index = int(np.floor(n/elmnt_types))
                hi_index = np.min( [int(np.floor(m/elmnt_types) + 2.0*support), self.N+1])
                lo_lim = mesh[lo_index]
                hi_lim = np.min( [ mesh[hi_index], mesh[-1]])
                basis_fn1 = funcs[n]
                basis_fn2 = funcs[m]
                func = lambda x: basis_fn1(x)*basis_fn2(x)
                val = quad(func, lo_lim, hi_lim, limit=100)[0]
                B[n,m] = B[m,n] = val
            #sys.stdout.write('Row: {0} of {1}\r'.format(n+1,dim))

        self.B = B


    # Compute Cholesky factorization of stiffness matrix
    def compute_chol_stiff(self):
        #print('\n   [ Computing Cholesky Factorization ]\n')
        A = self.A
        cholesky = cho_factor(A)
        self.cholesky_stiff = cholesky

    # Compute Cholesky factorization of mass matrix
    def compute_chol_mass(self):
        #print('\n   [ Computing Cholesky Factorization ]\n')
        B = self.B
        cholesky = cho_factor(B)
        self.cholesky_mass = cholesky

    # Compute Cholesky factorization of stiffness matrix
    def compute_chol(self):
        print('\n   [ Computing Cholesky Factorization ]\n\n')
        A = self.A
        B = self.B
        dt = self.dt
        K = B + dt*A
        cholesky = cho_factor(K)
        self.cholesky = cholesky


        
    # Project function into FEM space and return coefficients
    def proj_func(self, f):
        mesh = self.mesh
        dim = self.dim
        a = self.a
        b = self.b
        basis = self.basis.funcs
        support = self.basis.support
        support_overlap = self.basis.support_overlap
        elmnt_types = self.basis.elmnt_types
        cholesky = self.cholesky_mass

        rhs = np.zeros([dim,])
        for n in range(0,dim):
            lo_index = int(np.floor(n/elmnt_types))
            hi_index = np.min( [int(np.floor(n/elmnt_types) + 2.0*support), self.N+1])
            lo_lim = mesh[lo_index]
            hi_lim = np.min( [ mesh[hi_index], mesh[-1]])

            basis_fn = basis[n]
            func = lambda x: f(x)*basis_fn(x)
            val = quad(func, lo_lim, hi_lim, limit=100)[0]
            rhs[n] = val

        coeffs = cho_solve(cholesky, rhs)
        return coeffs
        

        
    # Form right-hand-side of weak formulation
    def compute_rhs(self, source):
        mesh = self.mesh
        dim = self.N
        dim = self.dim
        a = self.a
        b = self.b
        basis = self.basis.funcs
        support = self.basis.support
        support_overlap = self.basis.support_overlap
        elmnt_types = self.basis.elmnt_types

        rhs = np.zeros([dim,])
        for n in range(0,dim):
            lo_index = int(np.floor(n/elmnt_types))
            hi_index = np.min( [int(np.floor(n/elmnt_types) + 2.0*support), self.N+1])
            lo_lim = mesh[lo_index]
            hi_lim = np.min( [ mesh[hi_index], mesh[-1]])

            basis_fn = basis[n]
            func = lambda x: source(x)*basis_fn(x)
            val = quad(func, lo_lim, hi_lim, limit=100)[0]
            rhs[n] = val
            
        return rhs


    # Solve system using Cholesky factorization
    def solve_system(self):
        #print('\n   [ Solving Linear System ]\n')
        initial = self.initial
        source = self.source
        T = self.T

        self.soln_coeffs[:,0] = self.proj_func(initial)

        for t in range(0,T-1):
            self.time_step()
            sys.stdout.write('   [ Solving Linear System ]  (Step {0} of {1})\r'.format(t+1,T-1))

        print('\n\n')
        return self.soln_coeffs

    
    # Define time stepping scheme
    #  a_t+1  = (B + dt*A)^-1 * B * a_t  +  dt*(B + dt*A)^-1 * b_t+1
    def time_step(self):
        times = self.times
        t = self.current_step
        dt = self.dt

        B = self.B
        cholesky = self.cholesky
        
        alpha = self.soln_coeffs[:,t]

        current_source = lambda x: self.source(x,times[t+1])
        
        beta = self.compute_rhs(current_source)

        rhs = np.matmul(B,alpha) + dt*beta
        coeffs = cho_solve(cholesky, rhs)

        self.soln_coeffs[:,t+1] = coeffs
        self.current_step += 1
        
    
    # Evaluate function pointwise
    def evaluate_point(self, x, coeffs):
        val = 0.0
        dim = self.dim
        basis_funcs = self.basis.funcs
        
        for n in range(0,dim):
            basis_fn = basis_funcs[n]
            val = val + coeffs[n]*basis_fn(x)

        return val

    # Evaluate function on a mesh
    def evaluate_mesh(self, eval_mesh, times, coeffs):
        M = eval_mesh.size
        T = self.T
        vals = np.zeros([M,T])

        for t in range(0,T):
            for m in range(0,M):
                pt = eval_mesh[m]
                vals[m,t] = self.evaluate_point(pt, coeffs[:,t])

        return vals


        

        
        
