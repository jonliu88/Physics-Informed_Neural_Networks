
import gc
import numpy as np
import tensorflow as tf
import time
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras import layers, activations
import scipy.io
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
#!pip install tensorflow_addons
#import tensorflow_addons as tfa
#import json
#!pip -q install pyDOE
#from pyDOE import lhs  # for latin hypercube sampling
data = scipy.io.loadmat('./Aneurysm3D.mat')

t_star = data['t_star'] # T x 1
x_star = data['x_star'] # N x 1
y_star = data['y_star'] # N x 1
z_star = data['z_star'] # N x 1

T = t_star.shape[0]
N = x_star.shape[0]

U_star = data['U_star'] # N x T
V_star = data['V_star'] # N x T
W_star = data['W_star'] # N x T
P_star = data['P_star'] # N x T
C_star = data['C_star'] # N x T    

# Rearrange Data 
T_star = np.tile(t_star, (1,N)).T # N x T
X_star = np.tile(x_star, (1,T)) # N x T
Y_star = np.tile(y_star, (1,T)) # N x T
Z_star = np.tile(z_star, (1,T)) # N x T
T_data = T
N_data = N
idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
idx_x = np.random.choice(N, N_data, replace=False)
t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
z_data = Z_star[:, idx_t][idx_x,:].flatten()[:,None]
c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None]

T_eqns = T
N_eqns = N
idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
idx_x = np.random.choice(N, N_eqns, replace=False)
tcl = T_star[:, idx_t][idx_x,:].flatten()[:,None]
xcl = X_star[:, idx_t][idx_x,:].flatten()[:,None]
ycl = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
zcl = Z_star[:, idx_t][idx_x,:].flatten()[:,None]

t_data = t_data.astype(np.float32)
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)
z_data = z_data.astype(np.float32)
c_data = c_data.astype(np.float32)


tcl = tcl.astype(np.float32)
xcl = xcl.astype(np.float32)
ycl = ycl.astype(np.float32)
zcl = zcl.astype(np.float32)



def batch_pred():
    for i in range(T):
        out = u_PINN(tf.concat([np.reshape(X_star[:,i], (-1, 1)),np.reshape(Y_star[:,i], (-1, 1)),np.reshape(Z_star[:,i], (-1, 1)),np.reshape(T_star[:,i], (-1, 1))],1))
        if (i == 0):
            U_pred = tf.expand_dims(out[:,0],1)
            V_pred = tf.expand_dims(out[:,1],1)
            W_pred = tf.expand_dims(out[:,2],1)
            P_pred = tf.expand_dims(out[:,3],1)
            C_pred = tf.expand_dims(out[:,4],1)
        else:
            U_pred = tf.concat([U_pred,tf.expand_dims(out[:,0],1)],1)
            V_pred = tf.concat([V_pred,tf.expand_dims(out[:,1],1)],1)
            W_pred = tf.concat([P_pred,tf.expand_dims(out[:,2],1)],1)
            P_pred = tf.concat([P_pred,tf.expand_dims(out[:,3],1)],1)
            C_pred = tf.concat([C_pred,tf.expand_dims(out[:,4],1)],1)
    return U_pred, V_pred, W_pred, P_pred, C_pred

@tf.function
def loss(t_data_batch,x_data_batch,y_data_batch,z_data_batch,c_data_batch,tcl_batch,xcl_batch,ycl_batch,zcl_batch\
        ,weightsCL,weightsTr):
    #print(2)
    testOut = u_PINN(tf.concat([x_data_batch,y_data_batch,z_data_batch,t_data_batch],1))
    f1,f2,f3,f4,f5 = r_PINN(xcl_batch,ycl_batch,zcl_batch,tcl_batch)
    c_loss = tf.reduce_mean(tf.pow(tf.multiply(tf.subtract(c_data_batch,testOut[:,4:5]),weightsTr),2))
    #d_loss = tf.reduce_mean(tf.pow(tf.multiply(tf.subtract(tf.subtract(1.0,c_data_batch),testOut[:,5:6]),weightsDTr),2))
    r_loss = tf.reduce_mean(tf.pow(tf.multiply(weightsCL,f1),2) + tf.pow(tf.multiply(weightsCL,f2),2) + tf.pow(tf.multiply(weightsCL,f3),2) + tf.pow(tf.multiply(weightsCL,f4),2) + tf.pow(tf.multiply(weightsCL,f5),2)) #+ tf.pow(tf.multiply(weightsCL,f6),2))
    #print(5)
    return c_loss + r_loss
    
@tf.function
def grad(u_PINN,t_data_batch,x_data_batch,y_data_batch,z_data_batch,c_data_batch,tcl_batch,xcl_batch,ycl_batch,zcl_batch\
        ,weightsCL,weightsTr):
    #print(1)
    with tf.GradientTape(persistent=True) as tape:
        lossV = loss(t_data_batch,x_data_batch,y_data_batch,z_data_batch,c_data_batch,tcl_batch,xcl_batch,ycl_batch,zcl_batch\
        ,weightsCL,weightsTr)
        grads = tape.gradient(lossV, u_PINN.trainable_variables)
        gradCL = tape.gradient(lossV, weightsCL)
        gradTr = tape.gradient(lossV, weightsTr)
        #gradTr1 = tape.gradient(lossV, weightsTr)
        #gradTr2 = tape.gradient(lossV, weightsTr)
        #gradDTr = tape.gradient(lossV, weightsDTr)
    #print(6)
    #return lossV, grads, gradCL, gradTr#, gradDTr
    tf_optimizer.apply_gradients(zip(grads,u_PINN.trainable_variables))
    tf_optimizerW.apply_gradients(zip([-gradCL, -gradTr], [weightsCL, weightsTr]))
    
    
    
@tf.function
def r_PINN(x,y,z,t):
    #print(3)
    out = u_PINN(tf.concat([x,y,z,t],1))
    u = out[:,0:1]
    v = out[:,1:2]
    w = out[:,2:3]
    p = out[:,3:4]
    c = out[:,4:5]
    #d = out[:,5:6]
    
    u_t = tf.gradients(u,t)
    u_x = tf.gradients(u,x)
    u_y = tf.gradients(u,y)
    u_z = tf.gradients(u,z)
    u_xx = tf.gradients(u_x,x)
    u_yy = tf.gradients(u_y,y)
    u_zz = tf.gradients(u_z,z)
    
    v_t = tf.gradients(v,t)
    v_x = tf.gradients(v,x)
    v_y = tf.gradients(v,y)
    v_z = tf.gradients(v,z)
    v_xx = tf.gradients(v_x,x)
    v_yy = tf.gradients(v_y,y)
    v_zz = tf.gradients(v_z,z)
    
    w_t = tf.gradients(w,t)
    w_x = tf.gradients(w,x)
    w_y = tf.gradients(w,y)
    w_z = tf.gradients(w,z)
    w_xx = tf.gradients(w_x,x)
    w_yy = tf.gradients(w_y,y)
    w_zz = tf.gradients(w_z,z)
    
    c_t = tf.gradients(c,t)
    c_x = tf.gradients(c,x)
    c_y = tf.gradients(c,y)
    c_z = tf.gradients(c,z)
    c_xx = tf.gradients(c_x,x)
    c_yy = tf.gradients(c_y,y)
    c_zz = tf.gradients(c_z,z)
    
    #d_t = tf.gradients(d,t)
    #d_x = tf.gradients(d,x)
    #d_y = tf.gradients(d,y)
    #d_z = tf.gradients(d,z)
    #d_xx = tf.gradients(d_x,x)
    #d_yy = tf.gradients(d_y,y)
    #d_zz = tf.gradients(d_z,z)
    
    p_x = tf.gradients(p,x)
    p_y = tf.gradients(p,y)
    p_z = tf.gradients(p,z)
    
    val = np.float32(0.0101822)
    
    f1 = u_t + u*u_x + v*u_y + w*u_z + p_x - tf.multiply(val,(u_xx + u_yy + u_zz))
    f2 = v_t + u*v_x + v*v_y + w*v_z + p_y - tf.multiply(val,(v_xx + v_yy + v_zz))
    f3 = w_t + u*w_x + v*w_y + w*w_z + p_z - tf.multiply(val,(w_xx + w_yy + w_zz))
    f4 = u_x + v_y + w_z
    f5 = c_t + u*c_x + v*c_y + w*c_z - tf.multiply(val,(c_xx + c_yy + c_zz))
    #f6 = d_t + u*d_x + v*d_y + w*d_z - tf.multiply(val,(d_xx + d_yy + d_zz))
    #print(4)
    return f1 , f2 , f3 , f4 , f5 #, f6

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

layers = [4] + 10*[250] + [5]
u_PINN = neural_net(layers)
plot_ind = np.array([], dtype=int)
for iter in range(len(C_star)):
    if (z_star[iter]<15):
        plot_ind = np.append(plot_ind,iter)

tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008,beta_1=0.99)
tf_optimizerW = tf.keras.optimizers.Adam(learning_rate=0.0008,beta_1=0.99)

weightsCL = tf.Variable(tf.ones((10000,1),dtype='float32'))
weightsTr = tf.Variable(tf.ones((10000,1),dtype='float32'))

kern = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e4)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))
GP = GaussianProcessRegressor(kernel=kern,alpha=0,optimizer=None)
idx_data = np.random.choice(N*T, 10000)
idx_eqns = np.random.choice(N*T, 10000)
idx_W = np.random.choice(N*T, 10000)

(t_data_batch,
 x_data_batch,
 y_data_batch,
 z_data_batch,
 c_data_batch) = (t_data[idx_data,:],
                  x_data[idx_data,:],
                  y_data[idx_data,:],
                  z_data[idx_data,:],
                  c_data[idx_data,:])


(tcl_batch,
 xcl_batch,
 ycl_batch,
 zcl_batch) = (tcl[idx_eqns,:],
                  xcl[idx_eqns,:],
                  ycl[idx_eqns,:],
                  zcl[idx_eqns,:])
    
open('stokesBrainOutW.txt', 'w').close()
for iter in range(600001):
    if(iter!=0 and iter%500==0):
        with open('stokesBrainOutW.txt', 'a') as f:
            f.write("iter:" + str(iter))
            f.write("\n")
        X_train = np.concatenate((xcl_batch,ycl_batch,zcl_batch,tcl_batch),axis=1)
        y_train = weightsCL.numpy()[:,-1]
        GP.fit(X_train,y_train)
        idx_data = np.random.choice(N*T, 10000)
        idx_eqns = np.random.choice(N*T, 10000)



        (tcl_batch,
         xcl_batch,
         ycl_batch,
         zcl_batch) = (tcl[idx_eqns,:],
                          xcl[idx_eqns,:],
                          ycl[idx_eqns,:],
                          zcl[idx_eqns,:])
        weightsCL = tf.Variable(np.expand_dims(GP.predict(np.concatenate((xcl_batch,ycl_batch,zcl_batch,tcl_batch),axis=1)),-1),dtype=tf.float32)
        GP = None
        gc.collect()
        GP = GaussianProcessRegressor(kernel=kern,alpha=0,optimizer=None)
        X_train = np.concatenate((x_data_batch,y_data_batch,z_data_batch,t_data_batch,c_data_batch),axis=1)
        y_train = weightsTr.numpy()[:,-1]
        GP.fit(X_train,y_train)
        (t_data_batch,
         x_data_batch,
         y_data_batch,
         z_data_batch,
         c_data_batch) = (t_data[idx_data,:],
                          x_data[idx_data,:],
                          y_data[idx_data,:],
                          z_data[idx_data,:],
                          c_data[idx_data,:])
        weightsTr = tf.Variable(np.expand_dims(GP.predict(np.concatenate((x_data_batch,y_data_batch,z_data_batch,t_data_batch,c_data_batch),axis=1)),-1),dtype=tf.float32)
        GP = None
        gc.collect()
        GP = GaussianProcessRegressor(kernel=kern,alpha=0,optimizer=None)

    #idx_data = np.random.choice(N_data, 10000)
    #idx_eqns = np.random.choice(N_eqns, 10000)

    #(t_data_batch,
    # x_data_batch,
    # y_data_batch,
    # z_data_batch,
    # c_data_batch) = (t_data[idx_data,:],
    #                  x_data[idx_data,:],
    #                  y_data[idx_data,:],
    #                  z_data[idx_data,:],
    #                  c_data[idx_data,:])

    #print(idx_data[0:10])
    #print(t_data_batch[0:10])
    #print(x_data_batch[0:10])
    #print(y_data_batch[0:10])
    #print(z_data_batch[0:10])
    #print(c_data_batch[0:10])
    
    
    #(tcl_batch,
    # xcl_batch,
    # ycl_batch,
    # zcl_batch) = (tcl[idx_eqns,:],
    #                  xcl[idx_eqns,:],
    #                  ycl[idx_eqns,:],
    #                  zcl[idx_eqns,:])
    #print(tf.shape(t_data_batch))
    #for i in range(10):
        
    #lossV, grads, gradCL, gradTr = grad(u_PINN,t_data_batch,x_data_batch,y_data_batch,z_data_batch,c_data_batch,tcl_batch,xcl_batch,ycl_batch,zcl_batch\
    grad(u_PINN,t_data_batch,x_data_batch,y_data_batch,z_data_batch,c_data_batch,tcl_batch,xcl_batch,ycl_batch,zcl_batch\
                           ,weightsCL,weightsTr)
    tf.keras.backend.clear_session()
    #tf.compat.v1.reset_default_graph()
    #gradCL = None
    #gradTr = None
    #gc.collect()
    


    #print(out)
    if (iter%500 == 0):
    #    print("haha")
    #    print(iter)
        U_pred, V_pred, W_pred, P_pred, C_pred = batch_pred()
        #U_pred, C_pred = batch_pred()
        pPredErr = P_pred.numpy().T.flatten()
        pErr = P_star.T.flatten()
        pErr = pErr - (np.mean(pErr))
        pPredErr = pPredErr - (tf.reduce_mean(pPredErr))
        errU = np.linalg.norm(U_star.T.flatten()-U_pred.numpy().T.flatten(),2)/np.linalg.norm(U_star.T.flatten(),2)
        errV = np.linalg.norm(V_star.T.flatten()-V_pred.numpy().T.flatten(),2)/np.linalg.norm(V_star.T.flatten(),2)
        errW = np.linalg.norm(W_star.T.flatten()-W_pred.numpy().T.flatten(),2)/np.linalg.norm(W_star.T.flatten(),2)
        errP = np.linalg.norm(pErr-pPredErr,2)/np.linalg.norm(pErr,2)
        errC = np.linalg.norm(C_star.T.flatten()-C_pred.numpy().T.flatten(),2)/np.linalg.norm(C_star.T.flatten(),2)
        with open('stokesBrainOutW.txt', 'a') as f:
            f.write('U L2 error: %.4e' % (errU))
            f.write("\n")
            f.write('V L2 error: %.4e' % (errV))
            f.write("\n")
            f.write('W L2 error: %.4e' % (errW))
            f.write("\n")
            f.write('P L2 error: %.4e' % (errP))
            f.write("\n")
            f.write('C L2 error: %.4e' % (errC))
            f.write("\n")
    #    print('U L2 error: %.4e' % (errU))
    #    print('C L2 error: %.4e' % (errC))

        #print(lossV)
    if (iter %250 == 0):
        with open('stokesBrainOutW.txt', 'a') as f:
            f.write("iter:" + str(iter))
            f.write("\n")
        out = u_PINN(tf.concat([np.reshape(X_star[:,150:151], (-1, 1)),np.reshape(Y_star[:,150:151], (-1, 1)),np.reshape(Z_star[:,150:151], (-1, 1)),np.reshape(T_star[:,150:151], (-1, 1))],1))
        np.save('./numpy/numpyBrainUW' + str(iter),out[:,0:1].numpy()[plot_ind])
        np.save('./numpy/numpyBrainVW' + str(iter),out[:,1:2].numpy()[plot_ind])
        np.save('./numpy/numpyBrainWW' + str(iter),out[:,2:3].numpy()[plot_ind])
        np.save('./numpy/numpyBrainPW' + str(iter),out[:,3:4].numpy()[plot_ind])
        np.save('./numpy/numpyBrainCW' + str(iter),out[:,4:5].numpy()[plot_ind])