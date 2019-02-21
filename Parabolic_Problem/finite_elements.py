from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.special.orthogonal import p_roots
from scipy.integrate import quad, fixed_quad
from scipy.linalg import cho_factor, cho_solve
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import time
import sys

# Load basis functions
from basis_functions import *


#QUAD = 15

# Define Gaussian quadrature rule
#def gauss(f,a,b,quad_n):
#    [x,w] = p_roots(quad_n+1)
#    val = 0.0
#    for i in range(0,quad_n+1):
#        val = val + 0.5*(b-a)*w[i]*f(0.5*(b-a)*x[i] + 0.5*(b+a))
#    return val


# Defines one-dimensional Finite Element Space
class fem_space:

    # Initialize the class
    def __init__(self, mesh, times, basis_functions,
                 initial, source, stiffness=None,
                 time_stepping='Backward Euler', solve_method='Cholesky'):
        
        self.mesh = mesh
        self.N = mesh.size - 2
        self.a, self.b = [mesh[0], mesh[-1]]

        self.times = times
        self.T = times.size
        self.dt = times[1] - times[0]
        self.start_time, self.final_time = [times[0], times[-1]]
        
        self.current_step = 0

        self.basis = basis_functions()
        self.basis.funcs, self.basis.func_supports = self.init_funcs()
        self.basis.grads = self.init_grads()
        self.dim = self.basis.elmnt_types*self.N

        self.initial = initial
        self.source = source
        self.stiffness = stiffness

        self.time_stepping = time_stepping
        self.solve_method = solve_method

        self.soln_coeffs = np.zeros([self.dim, self.T])

        print('\n-------------------------')
        print(' Finite Element Space:  |')
        print('-------------------------\n')
        print('  - Spatial Domain = [%0.2f, %0.2f]' %(self.a,self.b))
        print('  - Time Interval = [%0.2f, %0.2f]' %(self.start_time,self.final_time))
        #print('  - Interior Nodes = %d' %(self.N))
        print('  - FEM Basis = %s' %(self.basis.name))
        print('  - Time Stepping = %s' %(self.time_stepping))
        print('  - Solve Method = %s' %(self.solve_method))
        print('  - Problem Dimension = %d\n' %(self.dim))

        print('\n-------------------------')
        print(' FEM Solver Progress:   |')
        print('-------------------------')


    # Initialize basis functions
    def init_funcs(self):
        mesh = self.mesh
        N = self.N
        support = self.basis.support
        elmnt_types = self.basis.elmnt_types
        funcs = []
        func_supports = []
        for n in range(0,N):
            for el_type in range(0,elmnt_types):
                func = lambda x, n=n, el_type=el_type: self.basis.func(x,mesh[n],mesh[n+support],el_type)
                func_support = [n, int(n + 2.0*support)]
                funcs.append(func)
                func_supports.append(func_support)
                
        return [funcs, func_supports]

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
        print('\n  [ Forming Stiffness Matrix ]\n')
        mesh = self.mesh
        dim = self.dim
        a,b = [self.a, self.b]
        func_supports = self.basis.func_supports
        grads = self.basis.grads
        support = self.basis.support
        support_overlap = self.basis.support_overlap
        elmnt_types = self.basis.elmnt_types

        stiffness = self.stiffness
        if stiffness:
            apply_stiff = True
        else:
            apply_stiff = False
            
        A = np.zeros([dim,dim])
        for n in range(0,dim):
            
            if n < dim-support_overlap:
                overlap = support_overlap
            else:
                overlap = (dim-n) - 1
                
            for m in range(n,n+overlap+1):
                lo_index = func_supports[n][0]
                hi_index = np.min([func_supports[m][1], self.N + 1])
                lo_lim = mesh[lo_index]
                hi_lim = np.min( [ mesh[hi_index], mesh[-1]])
                basis_grad_fn1 = grads[n]
                basis_grad_fn2 = grads[m]
                if apply_stiff:
                    func = lambda x: stiffness(x)*basis_grad_fn1(x)*basis_grad_fn2(x)
                else:
                    func = lambda x: basis_grad_fn1(x)*basis_grad_fn2(x)
                val = quad(func, lo_lim, hi_lim, limit=100)[0]
                #val = fixed_quad(func, lo_lim, hi_lim, n=QUAD)[0]
                #val = gauss(func, lo_lim, hi_lim, self.quad_n)
                A[n,m] = A[m,n] = val
            #sys.stdout.write('Row: {0} of {1}\r'.format(n+1,dim))

        self.A = A


    # Form mass matrix
    def form_mass(self):
        print('\n  [ Forming Mass Matrix ]\n')
        mesh = self.mesh
        dim = self.dim
        a,b = [self.a, self.b]
        funcs = self.basis.funcs
        func_supports = self.basis.func_supports
        support_overlap = self.basis.support_overlap
        
        B = np.zeros([dim,dim])
        for n in range(0,dim):
            if n < dim-support_overlap:
                overlap = support_overlap
            else:
                overlap = (dim-n) - 1
                
            for m in range(n,n+overlap+1):
                lo_index = func_supports[n][0]
                hi_index = np.min([func_supports[m][1], self.N + 1])
                lo_lim = mesh[lo_index]
                hi_lim = np.min( [ mesh[hi_index], mesh[-1]])
                basis_fn1 = funcs[n]
                basis_fn2 = funcs[m]
                func = lambda x: basis_fn1(x)*basis_fn2(x)
                val = quad(func, lo_lim, hi_lim, limit=100)[0]
                #val = fixed_quad(func, lo_lim, hi_lim, n=QUAD)[0]
                #val = gauss(func, lo_lim, hi_lim, self.quad_n)
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


    # Compute Cholesky factorization for time-stepping
    def compute_chol(self):
        print('\n  [ Computing Cholesky Factorization ]\n\n')
        A = self.A
        B = self.B
        dt = self.dt
        time_stepping = self.time_stepping

        if time_stepping == 'Backward Euler':
            K = B + dt*A
        elif time_stepping == 'Crank-Nicolson':
            K = B + 0.5*dt*A

        cholesky = cho_factor(K)
        self.cholesky = cholesky


    # Compute preconditioner for conjugate gradient method
    def compute_cg_precond(self):
        print('\n  [ Computing Preconditioner ]\n\n')
        time_stepping = self.time_stepping
        dim = self.dim
        A = self.A
        B = self.B
        dt = self.dt
        
        if time_stepping == 'Backward Euler':
            K = B + dt*A
        elif time_stepping == 'Crank-Nicolson':
            K = B + 0.5*dt*A
            
        sK = sparse.csc_matrix(K)
        precond_info = spla.spilu(sK)
        precond_x = lambda x: precond_info.solve(x)
        precond = spla.LinearOperator((dim,dim),precond_x)
        self.conj_grad = [sK, precond]


    # Project function into FEM space and return coefficients
    def proj_func(self, f):
        mesh = self.mesh
        dim = self.dim
        a,b = [self.a, self.b]
        basis = self.basis.funcs
        func_supports = self.basis.func_supports
        support_overlap = self.basis.support_overlap
        cholesky = self.cholesky_mass

        rhs = np.zeros([dim,])
        for n in range(0,dim):
            lo_index = func_supports[n][0]
            hi_index = np.min([func_supports[n][1], self.N + 1])
            lo_lim = mesh[lo_index]
            hi_lim = np.min( [ mesh[hi_index], mesh[-1]])

            basis_fn = basis[n]
            func = lambda x: f(x)*basis_fn(x)
            val = quad(func, lo_lim, hi_lim, limit=100)[0]
            #val = fixed_quad(func, lo_lim, hi_lim, n=QUAD)[0]
            #val = gauss(func, lo_lim, hi_lim, self.quad_n)
            rhs[n] = val

        coeffs = cho_solve(cholesky, rhs)
        return coeffs
        

        
    # Form right-hand-side of weak formulation
    def source_inner_products(self, source):
        mesh = self.mesh
        dim = self.N
        dim = self.dim
        a,b = [self.a, self.b]
        basis = self.basis.funcs
        func_supports = self.basis.func_supports
        support_overlap = self.basis.support_overlap

        inner_prods = np.zeros([dim,])
        for n in range(0,dim):
            lo_index = func_supports[n][0]
            hi_index = np.min([func_supports[n][1], self.N + 1])
            lo_lim = mesh[lo_index]
            hi_lim = np.min( [ mesh[hi_index], mesh[-1]])

            basis_fn = basis[n]
            func = lambda x: source(x)*basis_fn(x)
            val = quad(func, lo_lim, hi_lim, limit=100)[0]
            #val = fixed_quad(func, lo_lim, hi_lim, n=QUAD)[0]
            #val = gauss(func, lo_lim, hi_lim, self.quad_n)
            inner_prods[n] = val
            
        return inner_prods


    # Solve system using Cholesky factorization
    def solve_system(self):
        #print('\n   [ Solving Linear System ]\n')
        initial = self.initial
        source = self.source
        time_stepping = self.time_stepping
        solve_method = self.solve_method
        T = self.T

        self.soln_coeffs[:,0] = self.proj_func(initial)

        start_time = time.time()
        
        if solve_method == 'Cholesky':
            # Compute Cholesky factorization for time stepping
            self.compute_chol()
        elif solve_method == 'Conjugate Gradient':
            # Compute preconditioner for conjugate gradient
            self.compute_cg_precond()
            
        
        for t in range(0,T-1):
            if time_stepping == 'Backward Euler':
                self.time_step()
            elif time_stepping == 'Crank-Nicolson':
                self.time_step_cn()
            else:
                print('\nERROR: Unrecognized time-stepping scheme "%s" specified.' %(time_stepping))
            sys.stdout.write('  [ Solving Linear System ]  (Step {0} of {1})\r'.format(t+1,T-1))
            sys.stdout.flush()

        end_time = time.time()
        solve_time = end_time - start_time
        
        print('\n\n\n   Solve Time:  %.5f seconds' %(solve_time))
        print('\n')
        return self.soln_coeffs

    
    # Define time stepping scheme using Backward Euler
    #  a_t+1  = (B + dt*A)^-1 * B * a_t  +  dt*(B + dt*A)^-1 * b_t+1
    def time_step(self):
        solve_method = self.solve_method
        times = self.times
        t = self.current_step
        dt = self.dt

        B = self.B
        alpha = self.soln_coeffs[:,t]
        current_source = lambda x: self.source(x,times[t+1])
        beta = self.source_inner_products(current_source)
        rhs = np.matmul(B,alpha) + dt*beta

        if solve_method == 'Cholesky':
            cholesky = self.cholesky
            coeffs = cho_solve(cholesky, rhs)
        elif solve_method == 'Conjugate Gradient':
            cg_info = self.conj_grad
            K = cg_info[0]
            precond = cg_info[1]
            coeffs = spla.cg(K,rhs,M=precond)[0]


        self.soln_coeffs[:,t+1] = coeffs
        self.current_step += 1

    # Define time stepping scheme using Crank-Nicolson
    #  a_t+1  = (B + 1/2*dt*A)^-1 * (B - 1/2*dt*A) * a_t  +  dt*(B + 1/2*dt*A)^-1 * b_t+1/2
    def time_step_cn(self):
        solve_method = self.solve_method
        times = self.times
        t = self.current_step
        dt = self.dt

        A = self.A
        B = self.B
        
        alpha = self.soln_coeffs[:,t]
        t_half_step = 0.5*(times[t] + times[t+1])
        current_source = lambda x: self.source(x,t_half_step)
        beta = self.source_inner_products(current_source)
        rhs = np.matmul(B - 0.5*dt*A,alpha) + dt*beta

        if solve_method == 'Cholesky':
            cholesky = self.cholesky
            coeffs = cho_solve(cholesky, rhs)
        elif solve_method == 'Conjugate Gradient':
            cg_info = self.conj_grad
            K = cg_info[0]
            precond = cg_info[1]
            coeffs = spla.cg(K,rhs,M=precond)[0]

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


