from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
import sys


# Linear 'hat' functions for basis in one-dimension
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


# Quadratic functions for basis in one-dimension
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

