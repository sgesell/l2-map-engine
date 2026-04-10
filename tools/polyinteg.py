# -*- coding: utf-8 -*-
"""
Polynomial integration over arbitrary polygons using Gauss quadrature.

Created on Thu Aug 22 13:50:58 2019
@author: sgesell
"""

import numpy as np
from numba import njit


def polyf(coeff, x, y):
    """Evaluate 2D polynomial at point (x, y)."""
    v = [1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, 
         x**4, x**3*y, x**2*y**2, x*y**3, y**4]
    return sum(coeff*v)
    
    
def createpolys(X, Y):
    """Create basis polynomials for 9-point quadrilateral element.
    
    Args:
        X, Y: Coordinates of 9 integration points
        
    Returns:
        P: 9x15 matrix of polynomial coefficients (one row per basis function)
    """
    P = np.zeros((9, 15))
    K = np.array([[X[0]**2*Y[0]**2, X[0]*Y[0]**2, X[0]**2*Y[0], Y[0]**2, X[0]*Y[0], X[0]**2, Y[0], X[0], 1],
                  [X[1]**2*Y[1]**2, X[1]*Y[1]**2, X[1]**2*Y[1], Y[1]**2, X[1]*Y[1], X[1]**2, Y[1], X[1], 1],
                  [X[2]**2*Y[2]**2, X[2]*Y[2]**2, X[2]**2*Y[2], Y[2]**2, X[2]*Y[2], X[2]**2, Y[2], X[2], 1],
                  [X[3]**2*Y[3]**2, X[3]*Y[3]**2, X[3]**2*Y[3], Y[3]**2, X[3]*Y[3], X[3]**2, Y[3], X[3], 1],
                  [X[4]**2*Y[4]**2, X[4]*Y[4]**2, X[4]**2*Y[4], Y[4]**2, X[4]*Y[4], X[4]**2, Y[4], X[4], 1],
                  [X[5]**2*Y[5]**2, X[5]*Y[5]**2, X[5]**2*Y[5], Y[5]**2, X[5]*Y[5], X[5]**2, Y[5], X[5], 1],
                  [X[6]**2*Y[6]**2, X[6]*Y[6]**2, X[6]**2*Y[6], Y[6]**2, X[6]*Y[6], X[6]**2, Y[6], X[6], 1],
                  [X[7]**2*Y[7]**2, X[7]*Y[7]**2, X[7]**2*Y[7], Y[7]**2, X[7]*Y[7], X[7]**2, Y[7], X[7], 1],
                  [X[8]**2*Y[8]**2, X[8]*Y[8]**2, X[8]**2*Y[8], Y[8]**2, X[8]*Y[8], X[8]**2, Y[8], X[8], 1]])
    I = np.eye(9)
    for i, v in enumerate(I):
        p = np.linalg.solve(K, v)
        p = p[::-1]
        P[i, 0:6] = p[0:6]
        P[i, 7:9] = p[6:8]
        P[i, 12] = p[8]
    return P


@njit
def correctpoly2(coeff):
    """Split polynomial coefficients into homogeneous parts for degree-4 polynomial."""
    partsum = 15
    temp = np.zeros(int(partsum))
    temp[0:partsum] = coeff
    coeff = temp
    
    temp = np.zeros(1)
    C = [temp, temp, temp, temp, temp]
    for i in range(5):
        I = int((i) * (i+1) / 2)
        J = int((i+1) * (i+2) / 2)
        C[i] = coeff[I:J]
    return C, coeff


@njit
def gradp(x):
    """Helper function for polynomial degree calculation."""
    return -0.5 + np.sqrt(0.25 + 2*x) - 1


@njit
def combpoly(coeff1, coeff2):
    """Combine two polynomials by multiplication."""
    C1, coeff1 = correctpoly2(coeff1)
    C2, coeff2 = correctpoly2(coeff2)
    grd = (gradp(len(coeff1)) + gradp(len(coeff2))) + 1
    M = np.zeros((len(coeff2), int(grd * (grd+1) / 2)))
    C = np.outer(coeff2, coeff1)
#    print(C)
#    for i,c in enumerate(coeff2):
#        M[i,i+0:len(coeff1)+i]=C[i,:]
    I=-1
    for i,c in enumerate(C2):
        for j,x in enumerate(c):
            I=I+1
            sk=0
            for k,y in enumerate(C1):
#                print(str(I)+','+str(i)+','+str(j)+','+str(k)+','+str(sk))
                M[I,sk+I+k*i+0:(k+1)+I+k*i+sk]=C[I,sk+0:(k+1)+sk]
                sk=sk+k+1
    
    coeff=np.sum(M,axis=0)
#        for j,v in enumerate(ind):
##            M[i,i+v-1:i+v+j+1]=c[i,ind[j]:ind[j+1]]
#            print(c[i,ind[j]:ind[j+1]])
    return coeff

#@njit
def combpoly4(coeff1,coeff2):
    C1,coeff1=correctpoly2(coeff1)
    C2,coeff2=correctpoly2(coeff2)
    grd=(gradp(len(coeff1))+gradp(len(coeff2)))+1
    M=np.zeros((len(coeff2),int(grd*(grd+1)/2)))
    C=np.outer(coeff2,coeff1)
#    n=np.arange(0,len(C1)+1)
#    ind=(n*(n+1)/2).astype(int)
#    print(C)
#    for i,c in enumerate(coeff2):
#        M[i,i+0:len(coeff1)+i]=C[i,:]
   
    L=np.array([[1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	1,	0,	1,	1,	0,	1,	1,	1,	0,	1,	1,	1,	1,	0,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	1,	0,	1,	1,	0,	1,	1,	1,	0,	1,	1,	1,	1,	0,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	1,	0,	0,	1,	1,	0,	0,	1,	1,	1,	0,	0,	1,	1,	1,	1,	0,	0,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	1,	0,	0,	1,	1,	0,	0,	1,	1,	1,	0,	0,	1,	1,	1,	1,	0,	0,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	1,	0,	0,	1,	1,	0,	0,	1,	1,	1,	0,	0,	1,	1,	1,	1,	0,	0,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	1,	0,	0,	0,	1,	1,	1,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	1,	0,	0,	0,	1,	1,	1,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	1,	0,	0,	0,	1,	1,	1,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	1,	0,	0,	0,	1,	1,	1,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	1,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	1,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	1,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	1,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	1]])
    M[L==1]=C.flatten()
#        for j,v in enumerate(ind):
##            M[i,i+v-1:i+v+j+1]=c[i,ind[j]:ind[j+1]]
#            print(c[i,ind[j]:ind[j+1]])
    coeff=np.sum(M,axis=0)
    return coeff


@njit
def polinteg4(XY, coeff):
    """Integrate polynomial over arbitrary polygon using 5-point Gauss quadrature.
    
    Args:
        XY: Polygon vertices (n×2 array, first and last point should be same)
        coeff: Polynomial coefficients (degree 8 polynomial, 45 coefficients)
        
    Returns:
        INT: Integral value
    """
    d = 2
    V = XY[1:] - XY[0:-1]
    a = np.empty((len(V), 2))
    a[:, 0] = V[:, 1]
    a[:, 1] = -V[:, 0]
    b = np.empty((len(V), 1))
    b[:, 0] = [np.dot(a[i,:], XY[i,:]) for i in range(len(V))]
    
    # 5-point Gauss quadrature weights and points
    gp = np.array([0.04691007703066801815, 0.2307653449471585017, 0.5, 
                   0.7692346550528414983, 0.95308992296933192631])
    alpha = np.array([0.236927, 0.478629, 0.568889, 0.478629, 0.236927])
    
    # Split into homogeneous polynomial parts (degree 0 through 8)
    C = [coeff[0:1], coeff[1:3], coeff[3:6], coeff[6:10], coeff[10:15],
         coeff[15:21], coeff[21:28], coeff[28:36], coeff[36:45]]
    
    x = (np.outer(gp, (XY[1:, 0] - XY[:-1, 0])) + XY[:-1, 0]).T
    y = (np.outer(gp, (XY[1:, 1] - XY[:-1, 1])) + XY[:-1, 1]).T
    ab = np.outer(b, alpha)
    
    INT = 0.
    for i, v in enumerate(C):
        qu = 0
        for k, f in enumerate(v):
            if f:
                qu = qu + 0.5 * f * np.sum(np.sum(ab * x**(i-k) * y**k, axis=0), axis=0)
        INT = qu / (d + i) + INT
    
    return INT