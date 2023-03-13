import os
import argparse
from copy import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg as la
from funcs import *
import time 
ROOT_DIR = os.path.dirname(__file__)

def tgcyc(Ah,
          bh,
          u0,
          nsegment=64,
          engine=JOR,
          omega=.5,
          sigma=0.,
          operator="laplace", 
          weighting_scheme=None,
          iters=5,
          **kwargs):
    """
    Two grid cycle:
        - nsegment: the number of segment so that h = 1.0/nsegment
        - engine: the stationary iterative method used for smoothing

    Warning: make the good distinction between the number of segments, the
    number of nodes and the number of unknowns
    """
    if (nsegment % 2):
        raise ValueError("nsegment must be even")
    
    # Beware that : nsegment
    # n = number of nodes
    # n_inc = number of unknowns
    n = nsegment + 1
    n_inc_h = (nsegment-1) * (nsegment-1)
    n_inc_H = ((nsegment/2)-1) * ((nsegment/2)-1) 
    
    # build restriction operator R
    R = restriction2D(n_inc_h, weighting_scheme)
    
    # build prolongation operator P
    P = prolongation2D(n_inc_h, weighting_scheme)
    
    # plot_mat(R);
    AH = (R @ Ah) @ P
        
    # Pre-smoothing Relaxation
    uh, residual_history, uh_history = engine(Ah, bh, u0, omega=omega, maxiter=iters)
    
    #print(f"fine residual after {iters} iterations:", residual_history[-1]) 
    
    # compute the defect
    dh = np.ravel(bh - (Ah @ uh))

    # restriction
    dH = np.ravel(R @ dh)
    
    # solve linear system on coarse grid
    # eH, res_hist, _ = engine(AH, dH, np.zeros_like(dH), omega=omega, maxiter=100)
    eH = la.solve(AH, dH)
    
    # prolongation of error
    eh = np.ravel(P @ eH)
    # update approximation
    uh += eh
    
    uh_final, residual_history, uh_history = engine(Ah, bh, uh, omega=omega, maxiter=iters)
    iters = len(residual_history)
    plt.figure()
    plt.plot(np.arange(iters), residual_history, label="residual")
    plt.title("residual of tgcyc for postsmoothing")
    plt.xlabel("$iterations$")
    plt.ylabel("$\||residual\||$")
    plt.yscale("log")
    plt.show()
    
    return uh_final
    
    

# vcycle
def mgcyc(Ah,
          bh,
          u0,
          l=3,
          nsegment=64,
          engine=JOR,
          omega=.5,
          sigma=0.,
          operator="laplace", 
          weighting_scheme="half",
          iters=5,
          **kwargs):
    if (nsegment % 2**(l-1)):
         #
         raise ValueError("nsegment must be even")
    if (l<2):
         #
         raise ValueError("Unsufficient levels for v-mgcyc")
         
    # Beware that : nsegment
    # n = number of nodes
    # n_inc = number of unknowns
    n = nsegment + 1
    n_inc_h = (nsegment-1) * (nsegment-1)
    n_inc_H = ((nsegment/2)-1) * ((nsegment/2)-1) 
    
    # build restriction operator R
    R = restriction2D(n_inc_h, weighting_scheme)
    
    # build prolongation operator P
    P = prolongation2D(n_inc_h, weighting_scheme)

    AH = (R @ Ah) @ P
            
    # Pre-smoothing Relaxation
    uh, residual_history, uh_history = engine(Ah, bh, u0, omega=omega, maxiter=iters)

    # compute the defect    
    dh = np.ravel(bh - (Ah @ uh))

    # restriction with injection  
    dH = np.ravel(R @ dh) 
    
    if l == 2:
        # solve linear system with the lowest coarse grid
        eH = la.solve(AH, dH)
    else:
        eH = mgcyc(AH,dH,
                  np.zeros_like(dH),
                  l=l-1,
                  nsegment=(nsegment//2),
                  engine=engine,
                  omega=omega,
                  sigma=sigma,
                  operator=operator, 
                  weighting_scheme=weighting_scheme,
                  **kwargs)
     
    #post smoothing
    
    # prolongation of error
    eh = np.ravel(P @ eH)
    
    # update approximation
    uh += eh
     
    uh_final, residual_history, uh_history = engine(Ah, bh, uh, omega=omega, maxiter=iters)
    iters = len(residual_history)        
    plt.figure()
    plt.plot(np.arange(iters), residual_history, label="residual")
    plt.title("residual of mgcyc for postsmoothing")
    plt.xlabel("$iterations$")
    plt.ylabel("$\||residual\||$")
    plt.yscale("log")
    plt.show()
    
    return uh_final
   
# wcycle
def w_mgcyc(Ah,
          bh,
          u0,
          l=3,
          nsegment=64,
          engine=JOR,
          omega=.5,
          sigma=0.,
          operator="laplace", 
          weighting_scheme=None,
          iters=5,
          **kwargs):
    if (nsegment % 2**(l)):
         #
         raise ValueError("nsegment must be even")
    if (l<2):
         #
         raise ValueError("Unsufficient levels for w-mgcyc")
         
    # Beware that : nsegment
    # n = number of nodes
    # n_inc = number of unknowns
    n = nsegment + 1
    n_inc_h = (nsegment-1) * (nsegment-1)
    n_inc_H = ((nsegment/2)-1) * ((nsegment/2)-1) 
    
    # build restriction operator R
    R = restriction2D(n_inc_h, weighting_scheme)
    
    # build prolongation operator P
    P = prolongation2D(n_inc_h, weighting_scheme)
    
    AH = (R @ Ah) @ P
        
    # Pre-smoothing Relaxation
    uh, residual_history, uh_history = engine(Ah, bh, u0, omega=omega, maxiter=iters)

    # compute the defect    
    dh = np.ravel(bh - (Ah @ uh))

    # restriction with injection  
    dH = np.ravel(R @ dh) 
    
    if l == 3:
        uH = tgcyc(AH,dH,
          np.zeros_like(dH),
          nsegment=(nsegment//2),
          engine=engine,
          omega=omega,
          sigma=sigma,
          operator=operator, 
          weighting_scheme=weighting_scheme,
          **kwargs)
        eH = tgcyc(AH,dH,
          uH,
          nsegment=(nsegment//2),
          engine=engine,
          omega=omega,
          sigma=sigma,
          operator=operator, 
          weighting_scheme=weighting_scheme,
          **kwargs)
        
    else :
        
        uH = w_mgcyc(AH,dH,
                  np.zeros_like(dH),
                  l=l-1,
                  nsegment=(nsegment//2),
                  engine=engine,
                  omega=omega,
                  sigma=sigma,
                  operator=operator, 
                  weighting_scheme=weighting_scheme,
                  **kwargs)
        
        eH = w_mgcyc(AH, dH,
                  uH,#np.zeros_like(dH),
                  l=l-1,
                  nsegment=(nsegment//2),
                  engine=engine,
                  omega=omega,
                  sigma=sigma,
                  operator=operator, 
                  weighting_scheme=weighting_scheme,
                  **kwargs)

    #post smoothing
    
    # prolongation of error
    eh = np.ravel(P @ eH)
    
    # update approximation
    uh += eh
     
    uh_final, residual_history, uh_history = engine(Ah, bh, uh, omega=omega, maxiter=iters)    
    iters = len(residual_history)        
    plt.figure()
    plt.plot(np.arange(iters), residual_history, label="residual")
    plt.title("residual of w_mgcyc for postsmoothing")
    plt.xlabel("$iterations$")
    plt.ylabel("$\||residual\||$")
    plt.yscale("log")
    plt.show()

    return uh_final

def plot_mat(M):
    plt.figure(figsize=(5, 5))
    plt.imshow(M, interpolation='none', cmap='viridis')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    
# ====================================================================    
    # np.random.seed(0)
    operators = ["laplace", "anisotropic"]
    weighting_schemes = [None, "full", "half"]
    sigma = 1000000.
    omega = 1.5
    engine = SOR
    operator = operators[0]
    weighting_scheme = weighting_schemes[1]
    l=3 # number of levels for multigrid cycles
    nsegment = 2**5
    n = nsegment + 1
    h = 1.0 / nsegment
    
    n_inc_h = (nsegment-1) * (nsegment-1)
    # Full points
    xh = np.linspace(0., 1., n)
    # Inner points
    xih = xh[1:-1]
    yih = xh[1:-1]
    X_h, Y_h = np.meshgrid(xih, yih)
    # null rhs for now. Later try with random values
    # bh = np.zeros(n_inc_h).flatten()
    # bh = np.ones(n_inc_h).flatten()
    # bh = np.sin(X_h * pi)
    # bh = np.random.rand(n_inc_h)
    # bh[int(len(bh) * 0.25)] = 30
    # bh[int(len(bh) * 0.5)] = 20
    # bh[int(len(bh) * 0.75)] = 10
    bh = (np.sin(X_h * pi) + np.sin(3. * X_h * pi)) * \
    (np.sin(Y_h * pi) + np.sin(3. * Y_h * pi))
    bh = bh.reshape(n_inc_h,)
    
    # initial guess very high frequency. Try with different sines
    u0 = 0.5 * (np.sin(16 * X_h * pi) + np.sin(40. * X_h * pi)) * \
        (np.sin(16 * Y_h * pi) + np.sin(40. * Y_h * pi))    
    u0 = u0.flatten()
    
    
    # build an operator laplace/anisotropic
    Ah = operator2D(n_inc_h, sigma, h, operator)
    
    t_args = {
        "nsegment": nsegment,
        "engine": engine,
        "omega": omega,
        "operator": operator,
        "weighting_scheme": weighting_scheme,
        "u0": copy(u0),
        "bh": bh,
        "Ah": Ah
    }
    
    m_args = {
        "nsegment": nsegment,
        "engine": engine,
        "l": l,
        "omega": omega,
        "operator": operator,
        "weighting_scheme": weighting_scheme,
        "u0": copy(u0),
        "bh": bh,
        "Ah" :Ah
    }
    
    w_args = {
        "nsegment": nsegment,
        "engine": engine,
        "l": l,
        "omega": omega,
        "operator": operator,
        "weighting_scheme": weighting_scheme,
        "u0": copy(u0),
        "bh": bh,
        "Ah" :Ah
    }
    
    engine_args = {
        "A" :Ah,
        "b": bh,
        "x0": copy(u0),
        "omega": omega,
        "maxiter": 5000,
        "eps": 1e-20
    }   

# ===================================================================
    
#Generating informations for running multigrid functions
    #stats(engine, engine_args, tgcyc, t_args)
    #stats(engine, engine_args, mgcyc, m_args)
    stats(engine, engine_args, w_mgcyc, w_args)

    
    
    if True:
        
        # change the function that you want to plot
        grid_funcs_list = [tgcyc, mgcyc, w_mgcyc]
        grid_args_list = [t_args, m_args, w_args]
        for grid_func, grid_args in zip(grid_funcs_list, grid_args_list):
            u_grid = grid_func(**grid_args)
            fig, ax = plt.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d"}, figsize=(15, 8))
            ax[0].plot_surface(X_h, Y_h, u0.reshape(X_h.shape), color="blue", cmap=cm.coolwarm);
            ax[0].set_xlabel("$X$")
            ax[0].set_ylabel("$Y$")
            ax[0].set_zlabel("$U_{init}$")
            ax[0].set_title(f"Initial guess")
            ax[1].plot_surface(X_h, Y_h, u_grid.reshape(X_h.shape), color="red", cmap=cm.coolwarm);
            ax[1].set_xlabel("$X$")
            ax[1].set_ylabel("$Y$")
            ax[1].set_zlabel("$U_{grid}$")
            ax[1].set_title(f"Final approximation using {grid_func.__name__}")
            ax[2].plot_surface(X_h, Y_h, bh.reshape(X_h.shape), color="red", cmap=cm.coolwarm);
            ax[2].set_xlabel("$X$")
            ax[2].set_ylabel("$Y$")
            ax[2].set_zlabel("$B_{true}$")
            ax[2].set_title(f"True RHS for comparison")
