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
    def __init__(self, mesh, basis_functions):
        self.mesh = mesh
        self.a = mesh[0]
        self.b = mesh[-1]
        self.N = mesh.size - 2
        self.basis = basis_functions()

        self.dim = self.basis.elmnt_types*self.N
                
        self.basis.funcs = self.init_funcs()
        self.basis.grads = self.init_grads()

        print('\nFinite Element Space:  |')
        print('------------------------')
        print('\nInterval = [%0.2f,%0.2f]' %(self.a,self.b))
        print('Interior Nodes = %d' %(self.N))
        print('Basis = %s' %(self.basis.name))
        print('Problem Dimension = %d\n' %(self.dim))


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
        print('\n[ Forming Stiffness Matrix ]\n')
        mesh = self.mesh
        dim = self.dim
        a = self.a
        b = self.b
        grads = self.basis.grads
        support = self.basis.support
        support_overlap = self.basis.support_overlap
        elmnt_types = self.basis.elmnt_types
        
        K = np.zeros([dim,dim])
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
                K[n,m] = K[m,n] = val
            #sys.stdout.write('Row: {0} of {1}\r'.format(n+1,dim))

        #print('\nStiffness Matrix:\n')
        #print(K)
        #print('\n')
        #print(K[0:5,0:5])
        #print('\n')
        #print(K[-11:-6,-11:-6])
        #print('\n')
        #print(K[-6:-1,-6:-1])
        self.K = K

    # Compute Cholesky factorization of stiffness matrix
    def compute_chol(self):
        print('\n[ Computing Cholesky Factorization ]\n')
        K = self.K
        cholesky = cho_factor(K)
        self.cholesky = cholesky

    # Form right-hand-side of weak formulation
    def form_rhs(self, source):
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
            func = lambda x: -source(x)*basis_fn(x)
            val = quad(func, lo_lim, hi_lim, limit=100)[0]
            rhs[n] = val
            
        self.source = source
        self.rhs = rhs


    # Solve system using Cholesky factorization
    def solve_system(self):
        print('\n[ Solving Linear System ]\n\n')
        cholesky = self.cholesky
        rhs = self.rhs
        coeffs = cho_solve(cholesky, rhs)
        return coeffs

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
    def evaluate_mesh(self, eval_mesh, coeffs):
        M = eval_mesh.size
        vals = np.zeros([M,])

        for m in range(0,M):
            pt = eval_mesh[m]
            vals[m] = self.evaluate_point(pt, coeffs)

        return vals

    # Evaluate basis function on a mesh
    def evaluate_basis_mesh(self, eval_mesh, ID):
        M = eval_mesh.size
        vals = np.zeros([M,])
        funcs = self.basis.funcs
        basis_fn = funcs[ID]

        for m in range(0,M):
            pt = eval_mesh[m]
            vals[m] = basis_fn(pt)

        return vals

        
    # Evaluate basis gradient function on a mesh
    def evaluate_basis_grad_mesh(self, eval_mesh, ID):
        M = eval_mesh.size
        vals = np.zeros([M,])
        grads = self.basis.grads       
        basis_grad_fn = grads[ID]

        for m in range(0,M):
            pt = eval_mesh[m]
            vals[m] = basis_grad_fn(pt)

        return vals

        

        
        
