# import necessary libraries
print('test')
import tensorflow as tf
import numpy as np
import scipy.io
import math
#from tensorflow import keras
#from keras.models import Sequential
#from keras.layers import Dense, Input
#from keras import layers, activations
#from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import pyDOE
#from pyDOE import lhs

"""
Problem Definition and Quadrature Solution
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss

"""
This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez
    
    website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros

Much thanks goes to these individuals. It has been converted to Python by 
Abraham Lee.
"""

import numpy as np
from math import factorial

__all__ = ['lhs']

def lhs(n, samples=None, criterion=None, iterations=None):
    """
    Generate a latin-hypercube design
    
    Parameters
    ----------
    n : int
        The number of factors to generate samples for
    
    Optional
    --------
    samples : int
        The number of samples to generate for each factor (Default: n)
    criterion : str
        Allowable values are "center" or "c", "maximin" or "m", 
        "centermaximin" or "cm", and "correlation" or "corr". If no value 
        given, the design is simply randomized.
    iterations : int
        The number of iterations in the maximin and correlations algorithms
        (Default: 5).
    
    Returns
    -------
    H : 2d-array
        An n-by-samples design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.
    
    Example
    -------
    A 3-factor design (defaults to 3 samples)::
    
        >>> lhs(3)
        array([[ 0.40069325,  0.08118402,  0.69763298],
               [ 0.19524568,  0.41383587,  0.29947106],
               [ 0.85341601,  0.75460699,  0.360024  ]])
       
    A 4-factor design with 6 samples::
    
        >>> lhs(4, samples=6)
        array([[ 0.27226812,  0.02811327,  0.62792445,  0.91988196],
               [ 0.76945538,  0.43501682,  0.01107457,  0.09583358],
               [ 0.45702981,  0.76073773,  0.90245401,  0.18773015],
               [ 0.99342115,  0.85814198,  0.16996665,  0.65069309],
               [ 0.63092013,  0.22148567,  0.33616859,  0.36332478],
               [ 0.05276917,  0.5819198 ,  0.67194243,  0.78703262]])
       
    A 2-factor design with 5 centered samples::
    
        >>> lhs(2, samples=5, criterion='center')
        array([[ 0.3,  0.5],
               [ 0.7,  0.9],
               [ 0.1,  0.3],
               [ 0.9,  0.1],
               [ 0.5,  0.7]])
       
    A 3-factor design with 4 samples where the minimum distance between
    all samples has been maximized::
    
        >>> lhs(3, samples=4, criterion='maximin')
        array([[ 0.02642564,  0.55576963,  0.50261649],
               [ 0.51606589,  0.88933259,  0.34040838],
               [ 0.98431735,  0.0380364 ,  0.01621717],
               [ 0.40414671,  0.33339132,  0.84845707]])
       
    A 4-factor design with 5 samples where the samples are as uncorrelated
    as possible (within 10 iterations)::
    
        >>> lhs(4, samples=5, criterion='correlate', iterations=10)
    
    """
    H = None
    
    if samples is None:
        samples = n
    
    if criterion is not None:
        assert criterion.lower() in ('center', 'c', 'maximin', 'm', 
            'centermaximin', 'cm', 'correlation', 
            'corr'), 'Invalid value for "criterion": {}'.format(criterion)
    else:
        H = _lhsclassic(n, samples)

    if criterion is None:
        criterion = 'center'
    
    if iterations is None:
        iterations = 5
        
    if H is None:
        if criterion.lower() in ('center', 'c'):
            H = _lhscentered(n, samples)
        elif criterion.lower() in ('maximin', 'm'):
            H = _lhsmaximin(n, samples, iterations, 'maximin')
        elif criterion.lower() in ('centermaximin', 'cm'):
            H = _lhsmaximin(n, samples, iterations, 'centermaximin')
        elif criterion.lower() in ('correlate', 'corr'):
            H = _lhscorrelate(n, samples, iterations)
    
    return H

################################################################################

def _lhsclassic(n, samples):
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)    
    
    # Fill points uniformly in each interval
    u = np.random.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a
    
    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = np.random.permutation(range(samples))
        H[:, j] = rdpoints[order, j]
    
    return H
    
################################################################################

def _lhscentered(n, samples):
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)    
    
    # Fill points uniformly in each interval
    u = np.random.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    _center = (a + b)/2
    
    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        H[:, j] = np.random.permutation(_center)
    
    return H
    
################################################################################

def _lhsmaximin(n, samples, iterations, lhstype):
    maxdist = 0
    
    # Maximize the minimum distance between points
    for i in range(iterations):
        if lhstype=='maximin':
            Hcandidate = _lhsclassic(n, samples)
        else:
            Hcandidate = _lhscentered(n, samples)
        
        d = _pdist(Hcandidate)
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()
    
    return H

################################################################################

def _lhscorrelate(n, samples, iterations):
    mincorr = np.inf
    
    # Minimize the components correlation coefficients
    for i in range(iterations):
        # Generate a random LHS
        Hcandidate = _lhsclassic(n, samples)
        R = np.corrcoef(Hcandidate)
        if np.max(np.abs(R[R!=1]))<mincorr:
            mincorr = np.max(np.abs(R-np.eye(R.shape[0])))
            print('new candidate solution found with max,abs corrcoef = {}'.format(mincorr))
            H = Hcandidate.copy()
    
    return H
    
################################################################################

def _pdist(x):
    """
    Calculate the pair-wise point distances of a matrix
    
    Parameters
    ----------
    x : 2d-array
        An m-by-n array of scalars, where there are m points in n dimensions.
    
    Returns
    -------
    d : array
        A 1-by-b array of scalars, where b = m*(m - 1)/2. This array contains
        all the pair-wise point distances, arranged in the order (1, 0), 
        (2, 0), ..., (m-1, 0), (2, 1), ..., (m-1, 1), ..., (m-1, m-2).
    
    Examples
    --------
    ::
    
        >>> x = np.array([[0.1629447, 0.8616334],
        ...               [0.5811584, 0.3826752],
        ...               [0.2270954, 0.4442068],
        ...               [0.7670017, 0.7264718],
        ...               [0.8253975, 0.1937736]])
        >>> _pdist(x)
        array([ 0.6358488,  0.4223272,  0.6189940,  0.9406808,  0.3593699,
                0.3908118,  0.3087661,  0.6092392,  0.6486001,  0.5358894])
              
    """
    
    x = np.atleast_2d(x)
    assert len(x.shape)==2, 'Input array must be 2d-dimensional'
    
    m, n = x.shape
    if m<2:
        return []
    
    d = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            d.append((sum((x[j, :] - x[i, :])**2))**0.5)
    
    return np.array(d)

print('test')

# define grid for quadrature solution
utn = 128
uxn = 256
xlo = 0
xhi = 2.0
ux = np.linspace(xlo,xhi,uxn)
tlo = 0.0
thi = 4.0
ut = np.linspace (tlo,thi,utn)

# compute solution u(x,t) by quadrature of analytical formula:
u_quad = np.zeros([uxn,utn])
for utj in range(utn):
  if (ut[utj]==0.0):
    for uxj in range(uxn):
      u_quad[uxj,utj] = (ux[uxj]*(2-ux[uxj]))
  else:
    for uxj in range(uxn):
      for n in range(1,31):

        u_quad[uxj,utj] = u_quad[uxj,utj] + (-(16*((-1)**(n))-16)/(((np.pi)**3)*(n**3)))*(np.sin(n*np.pi*ux[uxj]/2))*np.exp(-.3*((n*np.pi/2)**2)*ut[utj])

# flatten grid and solution
X,T = np.meshgrid(ux,ut)
X_flat = tf.convert_to_tensor(np.hstack((X.flatten()[:,None],T.flatten()[:,None])),dtype=tf.float32)
u_flat = u_quad.T.flatten()
print('test')
def loss(xcl,tcl,x0,t0,u0,xlb,tlb,ulb,xub,tub,uub):
    predict0 = u_PINN(tf.concat([x0, t0], 1))
    predictL = u_PINN(tf.concat([xlb, tlb], 1))
    predictU = u_PINN(tf.concat([xub, tub], 1))
    residual = r_PINN(xcl, tcl)

    initialLoss = tf.reduce_mean(tf.pow(tf.subtract(predict0,u0),2))
    lowerLoss = tf.reduce_mean(tf.pow(tf.subtract(predictL,ulb),2))
    upperLoss = tf.reduce_mean(tf.pow(tf.subtract(predictU,uub),2))
    residualLoss = tf.reduce_mean(tf.pow(residual,2))

    return initialLoss + lowerLoss + upperLoss + residualLoss

@tf.function
def grad(model,xcl,tcl,x0,t0,u0,xlb,tlb,ulb,xub,tub,uub):
    with tf.GradientTape(persistent=True) as tape:
     lossV = loss(xcl,tcl,x0,t0,u0,xlb,tlb,ulb,xub,tub,uub)
     gradV = tape.gradient(lossV, model.trainable_variables)
    return lossV, gradV
number = 10000
randomPoints = lhs(2, number)
xcl = tf.expand_dims(tf.convert_to_tensor(xlo+(xhi-xlo)*randomPoints[:,0],dtype=tf.float32),-1)
tcl = tf.expand_dims(tf.convert_to_tensor(tlo+(thi-tlo)*randomPoints[:,1],dtype=tf.float32),-1)

number = 500
randomPoints = lhs(1, number)
x0 = tf.expand_dims(tf.convert_to_tensor(xlo+(xhi-xlo)*randomPoints[:,0],dtype=tf.float32),-1)
t0 = tf.expand_dims(tf.zeros(number),-1)
u0 = ((x0)*(2-x0))


number = 500
randomPoints = lhs(1, number)
tub = tf.expand_dims(tf.convert_to_tensor(tlo+(thi-tlo)*randomPoints[:,0],dtype=tf.float32),-1)
randomPoints = lhs(1, number)
tlb = tf.expand_dims(tf.convert_to_tensor(tlo+(thi-tlo)*randomPoints[:,0],dtype=tf.float32),-1)
xub = xhi*tf.ones(tf.shape(tub), dtype=tf.float32)
xlb = xlo*tf.ones(tf.shape(tub), dtype=tf.float32)
uub = tf.zeros(tf.shape(tub),dtype=tf.float32)
ulb = tf.zeros(tf.shape(tub),dtype=tf.float32)

# solution neural network
def neural_net(layer_sizes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(tf.keras.layers.Dense(
            width, tf.nn.tanh,
            kernel_initializer="glorot_normal"))
    model.add(tf.keras.layers.Dense(
            layer_sizes[-1], None,
            kernel_initializer="glorot_normal"))
    return model
# training loop

# initialize new instance of NN
layer_sizes = [2] + 8*[20] + [1]
u_PINN = neural_net(layer_sizes)
print('test')
# residual neural network
@tf.function
def r_PINN(x, t):
    u = u_PINN(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    return u_t - tf.multiply(u_xx, 0.3)

# Adam optimizer
tf_optimizer = tf.keras.optimizers.Adam(0.003,0.99)

for iter in range(4000):
  
  # compute gradients using AD
  loss_value,grads = grad(u_PINN,xcl,tcl,x0,t0,u0,xlb,tlb,ulb,xub,tub,uub)
    
  # update neural network weights
  tf_optimizer.apply_gradients(zip(grads,u_PINN.trainable_variables))
  
  # display intermediate results
  if ((iter + 1) % 200 == 0):
    print('iter =  '+str(iter+1))
    print('loss = {:.4f}'.format(loss_value))
    u_PINN_flat = u_PINN(X_flat)
    err = np.linalg.norm(u_flat-u_PINN_flat[:,-1],2)/np.linalg.norm(u_flat,2)
    print('L2 error: %.4e' % (err))
    fig = plt.figure(figsize=(12,4),dpi=75)
    plt.style.use('seaborn')
    for gi,snap in enumerate([0,0.15,0.5,0.85]):
      tind = int(snap*len(ut))
      ax = fig.add_subplot(1,4,gi+1)
      ax.set_aspect(0.5)
      ax.plot(ux,u_flat[tind*uxn:(tind+1)*uxn],'b-',linewidth=2,label='Exact')       
      ax.plot(ux,u_PINN_flat[tind*uxn:(tind+1)*uxn,0],'r--',linewidth=2,label='Prediction')
      ax.set_title('$t = %.2f$' % (ut[tind]),fontsize=10)
      ax.set_xlabel('$x$')
      ax.set_ylim([-1.3,1.3])
    plt.show()
    fig.savefig('./figures/heat' + str(iter))