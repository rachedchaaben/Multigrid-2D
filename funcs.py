from copy import copy
from math import pi, sin, cos, sqrt, floor
#import pylab as plt
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy as sp
import scipy.sparse as spa
import scipy.linalg as la
# from matplotlib import rc, rcParams
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })
# Some paramters
_eps =1e-12
_maxiter=500

def _basic_check(A, b, x0):
    """ Common check for clarity """
    n, m = A.shape
    if(n != m):
        raise ValueError("Only square matrix allowed")
    if(b.size != n):
        raise ValueError("Bad rhs size")
    if (x0 is None):
        x0 = np.zeros(n)
    if(x0.size != n):
        raise ValueError("Bad initial value size")
    return x0

def JOR(A, b, x0=None, omega=0.5, eps=_eps, maxiter=_maxiter):
    """
    Methode itérative stationnaire de sur-relaxation (Jacobi over relaxation)
    Convergence garantie si A est à diagonale dominante stricte
    A = D - E - F avec D diagonale, E (F) tri inf. (sup.) stricte
    Le préconditionneur 
    Output:
        - x is the solution at convergence or after maxiter iteration
        - residual_history is the norm of all residuals
    """
    if (omega > 1.) or (omega < 0.):
        raise ArithmeticError("JOR will diverge")
        
    x = _basic_check(A, b, x0)
    r = np.zeros(x.shape)
    residual_history = list()
    x_history = list()
    x_history.append(copy(x))
    i = 0
    M = 1/omega * np.diag(A)
    r = b - (A @ x)
    
    while (i < maxiter and la.norm(r) > eps):
        r = b - (A @ x)
        x += r / M
        residual_history.append(la.norm(r))
        x_history.append(copy(x))
        i += 1
        
    return x, residual_history, x_history

def SOR(A, b, x0=None, omega=1.5, eps=_eps, maxiter=_maxiter):
    """
    Methode itérative stationnaire de sur-relaxation successive
    (Successive Over Relaxation)
    A = D - E - F avec D diagonale, E (F) tri inf. (sup.) stricte
    Le préconditionneur est tri. inf. M = (1./omega) * D - E
    * Divergence garantie pour omega <= 0. ou omega >= 2.0
    * Convergence garantie si A est symétrique définie positive pour
    0 < omega  < 2.
    * Convergence garantie si A est à diagonale dominante stricte pour
    0 < omega  <= 1.
    Output:
        - x is the solution at convergence or after maxiter iteration
        - residual_history is the norm of all residuals
    """
    if (omega > 2.) or (omega < 0.):
        raise ArithmeticError("SOR will diverge")
    
    x = _basic_check(A, b, x0)
    r = np.zeros(x.shape)
    residual_history = list()
    x_history = list()
    x_history.append(copy(x))
    i = 0
    # M = (1/omega * np.diag(A)) - np.tril(A)
    # in the slides, M = (1/omega * diag(A) - E) where E is lower triangular matrix of A
    # and then update x = x + (M^-1 @ r)
    # Therefore, M @ y = r becomes y = r/M
    # we solve the linear system for y in the following loop
    M = np.tril(A)
    M[np.diag_indices(A.shape[0])] /= omega
    r = b - (A @ x)
    
    while (i < maxiter and la.norm(r) > eps):
        
        r = b - (A @ x)
        x += la.solve_triangular(M, r, lower=True)
        residual_history.append(la.norm(r))
        x_history.append(copy(x))
        i += 1
        
    return x, residual_history, x_history

def operator2D(n_, sigma, h, which="laplace"):
    
    """
    Returns an operator for 2D poisson equation 1st(laplace) or 2nd(anisotropic) equation
    It's a block tridiagonal matrix. Look at lecture 3 page 6 in pdf
    """
    
    if sigma < 0:
        raise ArithmeticError("sigma cannot be negative")
    
    n_ = floor(n_)
    A = np.zeros((n_, n_))
    
    n = int(sqrt(n_))
    h_sq = h**2

    I = np.identity(n)
    
    if which == "laplace":
        diags = [-1 * np.ones(n-1),
                 (4+sigma*h_sq) * np.ones(n),
                 -1 * np.ones(n-1)]
    
    elif which == "anisotropic":
        epsilon = sigma
        diags = [-1 * epsilon * np.ones(n-1),
                 2 * (1+epsilon) * np.ones(n),
                 -1 * epsilon * np.ones(n-1)]
    else:
        raise ValueError("given invalid operator type")
    
    B = spa.diags(diags, [-1, 0, 1]).toarray()
    
    
    for i in range(n):
        st = i * n
        end = (i+1) * n
        A[st:end, st:end] = B
        if i < n-1:
            end2 = (i+2) * n
            A[st:end, end:end2] = I
            A[end:end2, st:end] = I
        
    return h_sq * A

def restriction1D(n_, weighting_scheme="full"):
    """
    restriction operator: fine to coarse
    """
    n = int(n_)
    R = np.zeros((n // 2, n))
    
    vals = np.array([1, 2, 1])
    factor = 0.25
    # in 1D there is no half weighting only none and full
    # I have kept half == full for 1D to keep it consistent with 2D
    if weighting_scheme == None:
        vals -= 1
        factor = 1
        
    for i in range(n // 2):
        j = 2 * i
        if j + 3 <= n:
            R[i, j:j+3] = vals
        else:
            R[i, j:n] = vals[:(n-j)]
    
    return factor * R

def restriction2D(n_, weighting_scheme=None):
    """
    restriction operator: fine to coarse
    """
    # basic injection
    R_1D = restriction1D(floor(sqrt(n_)), weighting_scheme)
    
    if weighting_scheme != None:
        R_1D *= 4

    R = np.kron(R_1D, R_1D) 
    # weights from 4 neighbours
    if weighting_scheme == "half":
        mask = (R == 1) | (R == 2)
        R[mask] = R[mask] - 1
        R = 0.125 * R
    # weights from 8 neighbours
    elif weighting_scheme == "full":
        R = 0.0625 * R
    
    return R

def prolongation1D(n_, weighting_scheme):
    """
    prolongation or linear interpolation: coarse to fine
    """
    if weighting_scheme == None:
        factor = 1
    else: # in 1D there is no half weighting only none and full
    # I have kept half == full for 1D to keep it consistent with 2D
        factor = 2
    return np.transpose(restriction1D(n_, weighting_scheme)) * factor

def prolongation2D(n_, weighting_scheme):
    """
    prolongation or linear interpolation: coarse to fine
    """
    
    if weighting_scheme == None:
        factor = 1
    elif weighting_scheme == "half":
        factor = 2
    else:
        factor = 4
    return np.transpose(restriction2D(n_, weighting_scheme)) * factor

def l2norm(x, y):
    return np.sqrt(np.sum((x.ravel() - y.ravel())**2))

def stats(engine, engine_args, grid_func, grid_args):
    
    from time import time_ns

    st = time_ns()
    # iterative solver applied with large number of iterations
    u_iter, res_hist, _ = engine(**engine_args)
    iter_t = time_ns() - st
    iters = len(res_hist)
    del st

    # grid method
    st = time_ns()
    u_grid = grid_func(**grid_args)
    grid_t = time_ns() - st
    del st
    
    # direct solver
    Ah = grid_args["Ah"]
    u0 = grid_args["u0"]
    bh = grid_args["bh"]
    
    st = time_ns()
    r = bh - np.ravel(la.solve(Ah, u0))
    direct_t =  time_ns() - st
    r_grid = bh -Ah@u_grid
    r_engine =bh - Ah@u_iter
    
    print()
    print("="*40)
    print(f"direct solver\ttime={direct_t/10e6:.2f} sec residual norm: {la.norm(r):e}")
    print(f"{grid_func.__name__}\ttime={grid_t/10e6:.2f} residual norm: {la.norm(r_grid):e}")
    print(f"{engine.__name__}(i={iters})\ttime={iter_t/10e6:.2f} sec residual norm: {la.norm(r_engine):e}")
    print(f"l2 norm (vs iterative): {l2norm(u_iter, u_grid):e}")
    print("="*40, end="\n\n")
