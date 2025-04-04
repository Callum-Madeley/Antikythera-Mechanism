import pandas as pd
import numpy as np
import os

# jax for automatic differentiation
import jax
import jax.numpy as jnp

######################################################################################

# model to return cartesian coordinates (X,Y) and phase phi 

def modelxy(N,R,a0,a1,a2,a3,a4,a5,a6,a7,xc0,xc1,xc2,xc3,xc4,xc5,xc6,xc7,yc0,yc1,yc2,yc3,yc4,yc5,yc6,yc7):

    
    sizes = [1,22,9,37,1,3,2,6] # no. points in each section    
    
    xc = [xc0,xc1,xc2,xc3,xc4,xc5,xc6,xc7]
    yc = [yc0,yc1,yc2,yc3,yc4,yc5,yc6,yc7]
    a = [a0,a1,a2,a3,a4,a5,a6,a7]

    # duplicate values of xc, yc, alpha, to make array of length 81 (matching no. points) with values corresponding to each section
    # this is equivalent to iterating over [i] which doesn't work in this JAX-numpyro format

    xc_expand = jnp.array([val for val, count in zip(xc, sizes) for _ in range(count)])
    yc_expand = jnp.array([val for val, count in zip(yc, sizes) for _ in range(count)])
    a_expand = jnp.array([val for val, count in zip(a, sizes) for _ in range(count)])
    
    # index points 1-81 with i = [0,80]
    index = jnp.linspace(0,80,81)

    # phase offset for each point from start of its section
    phi = 2 * jnp.pi * index/N  +  (a_expand * jnp.pi/180) # degrees --> radians

    # model (x,y) coordinates
    xi = (R * jnp.cos(phi) + xc_expand)
    yi = (R * jnp.sin(phi) + yc_expand)
    

    return xi,yi,phi

#####################################################################################

# negative log likelihood function for isotropic model

def nll_isotropic(params,data):

    # unpack parameters:
    N,R, a0,a1,a2,a3,a4,a5,a6,a7,sigma, xc0,xc1,xc2,xc3,xc4,xc5,xc6,xc7,yc0,yc1,yc2,yc3,yc4,yc5,yc6,yc7 = params # note sigma_x = sigma_y so just one error parameter

    # unpack data
    X, Y = data

    # extract model xi,yi

    xi, yi, phi = modelxy(R,N,xc0,xc1,xc2,xc3,xc4,xc5,xc6,xc7,yc0,yc1,yc2,yc3,yc4,yc5,yc6,yc7,a0,a1,a2,a3,a4,a5,a6,a7)

    # likelihood on x & y

    x_likelihood = - np.sum((xi - X)**2 / (2 * sigma**2))
    y_likelihood = - np.sum((yi - Y)**2 / (2 * sigma**2))

    # prefactor
    norm = - len(X) * np.log(2 * np.pi * sigma * sigma)

    # total likelihood

    likelihood_isotropic = norm + np.sum(x_likelihood + y_likelihood)

    return likelihood_isotropic



####################################################################################


# negative log likelihood function for radial-tangential model

def nll_aligned(params,data):

    # unpack parameters:
    N,R, a0,a1,a2,a3,a4,a5,a6,a7,sigma_r, sigma_t, xc0,xc1,xc2,xc3,xc4,xc5,xc6,xc7,yc0,yc1,yc2,yc3,yc4,yc5,yc6,yc7 = params # note sigma_r =! sigma_t ONE MORE ERROR PARAMETER HERE

    # unpack data
    X, Y = data

    # extract model xi,yi

    xi, yi, phi = modelxy(R,N,xc0,xc1,xc2,xc3,xc4,xc5,xc6,xc7,yc0,yc1,yc2,yc3,yc4,yc5,yc6,yc7,a0,a1,a2,a3,a4,a5,a6,a7)

    # determine error between data & model (easier to do before coordinate shift)

    e_x = xi - X 
    e_y = yi - Y

    # project error into radial & tangential directions with coordinate transform depending on phi values
    r_p = e_x * np.cos(phi) + e_y * np.sin(phi)
    t_p = e_x * np.sin(phi) - e_y * np.cos(phi)

    # likelihood on radial & tangential components (errors independent so sum is separable)

    r_likelihood = - np.sum(r_p**2 / (2 * sigma_r**2))
    t_likelihood = - np.sum(t_p**2 / (2 * sigma_t**2))

    # prefactor
    norm = - len(X) * np.log(2 * np.pi * sigma_r*sigma_t)

    # total likelihood

    likelihood_aligned = norm + r_likelihood + t_likelihood

    return likelihood_aligned



    
