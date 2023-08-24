import numpy as np
import tensorflow as tf
import time
#import gdown
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras import layers, activations
import scipy.io
from scipy.interpolate import griddata
#!pip install tensorflow_addons
#import tensorflow_addons as tfa
import json
#!pip -q install pyDOE
#from pyDOE import lhs  # for latin hypercube sampling
file = open('Cylinder2D_C_rectangular.json','r')
C_rect = np.array(json.load(file))
file.close()
file = open('Cylinder2D_U_rectangular.json','r')
U_rect = np.array(json.load(file))
file.close()
file = open('Cylinder2D_V_rectangular.json','r')
V_rect = np.array(json.load(file))
file.close()
file = open('Cylinder2D_P_rectangular.json','r')
P_rect = np.array(json.load(file))
file.close()
file = open('Cylinder2D_x_rectangular.json','r')
x_rect = np.array(json.load(file))
file.close()
file = open('Cylinder2D_y_rectangular.json','r')
y_rect = np.array(json.load(file))
file.close()
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

NS = C_rect.shape[0] # number of spatial points
NT = C_rect.shape[1] # number of time points

t_rect = np.linspace(0,16,NT)

# inlet boundary condition points
# sample Nb points from len(idb)*NT = 20301 total points
Nb = 20000
idb = np.where(x_rect==x_rect.min())[0]
idbs = np.random.choice(len(idb)*NT,Nb,replace=False)
xb = tf.expand_dims(tf.convert_to_tensor(np.tile(x_rect[idb],NT)[idbs],dtype=tf.float32),axis=1)
yb = tf.expand_dims(tf.convert_to_tensor(np.tile(y_rect[idb],NT)[idbs],dtype=tf.float32),axis=1)
tb = tf.expand_dims(tf.convert_to_tensor(np.tile(t_rect,(len(idb),1)).T.flatten()[idbs],dtype=tf.float32),axis=1)
ub = tf.expand_dims(tf.convert_to_tensor(U_rect[idb,:].T.flatten()[idbs],dtype=tf.float32),axis=1)
vb = tf.expand_dims(tf.convert_to_tensor(V_rect[idb,:].T.flatten()[idbs],dtype=tf.float32),axis=1)

# collocation points
# must remove points in the cylinder region (x^2+y^2<=0.5^2)
# this removes about 1.57% of the points
Ncl = 100000
idcl = np.random.choice(NS*NT,Ncl,replace=False)
xcl = tf.expand_dims(tf.convert_to_tensor(np.tile(x_rect,NT)[idcl],dtype=tf.float32),axis=1)
ycl = tf.expand_dims(tf.convert_to_tensor(np.tile(y_rect,NT)[idcl],dtype=tf.float32),axis=1)
tcl = tf.expand_dims(tf.convert_to_tensor(np.tile(t_rect,(NS,1)).T.flatten()[idcl],dtype=tf.float32),axis=1)

# concentration training data points
# sample Ntr points from NS*NT = 4017588
Ntr = 100000
idtr = np.random.choice(NS*NT,Ntr,replace=False)
xtr = tf.expand_dims(tf.convert_to_tensor(np.tile(x_rect,NT)[idtr],dtype=tf.float32),axis=1)
ytr = tf.expand_dims(tf.convert_to_tensor(np.tile(y_rect,NT)[idtr],dtype=tf.float32),axis=1)
ttr = tf.expand_dims(tf.convert_to_tensor(np.tile(t_rect,(NS,1)).T.flatten()[idtr],dtype=tf.float32),axis=1)
ctr = tf.expand_dims(tf.convert_to_tensor(C_rect.T.flatten()[idtr],dtype=tf.float32),axis=1)

# concentration testing data points
# points not selected for training
idts =  np.setdiff1d(np.arange(NS*NT),idtr,assume_unique=True)
xts = tf.expand_dims(tf.convert_to_tensor(np.tile(x_rect,NT)[idts],dtype=tf.float32),axis=1)
yts = tf.expand_dims(tf.convert_to_tensor(np.tile(y_rect,NT)[idts],dtype=tf.float32),axis=1)
tts = tf.expand_dims(tf.convert_to_tensor(np.tile(t_rect,(NS,1)).T.flatten()[idts],dtype=tf.float32),axis=1)
cts = tf.expand_dims(tf.convert_to_tensor(C_rect.T.flatten()[idts],dtype=tf.float32),axis=1)
#T,X = np.meshgrid(t_rect,x_rect)
#X_flat = tf.expand_dims(X.T.flatten(),axis=1)
#T_flat = tf.expand_dims(T.T.flatten(),axis=1)
#U_flat = U_rect.T.flatten()
#T,Y = np.meshgrid(t_rect,y_rect)
#Y_flat = tf.expand_dims(Y.T.flatten(),axis=1)
#print(tf.shape(X_flat))
#print(tf.shape(Y_flat))
#print(tf.shape(T_flat))


def batch_pred():
    i = 1
    # batch prediction due to large size
    x = tf.expand_dims(tf.convert_to_tensor(x_rect,dtype=tf.float32),1)
    y = tf.expand_dims(tf.convert_to_tensor(y_rect,dtype=tf.float32),1)
    for i in range(NT):
      t = tf.convert_to_tensor(t_rect[i]*np.ones([NS,1]),dtype=tf.float32)
      out  = u_PINN(tf.concat([x,y,t],1)) # u,v,p,c prediction for at time t
      if i==0:
        U_pred = tf.expand_dims(out[:,0],1)
        V_pred = tf.expand_dims(out[:,1],1)
        P_pred = tf.expand_dims(out[:,2],1)
        C_pred = tf.expand_dims(out[:,3],1)
      else:
        U_pred = tf.concat([U_pred,tf.expand_dims(out[:,0],1)],1)
        V_pred = tf.concat([V_pred,tf.expand_dims(out[:,1],1)],1)
        P_pred = tf.concat([P_pred,tf.expand_dims(out[:,2],1)],1)
        C_pred = tf.concat([C_pred,tf.expand_dims(out[:,3],1)],1)
    return U_pred, V_pred, P_pred, C_pred
step = 0.05  # grid quantization

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
def get_imgseq(D):
    """
    - Normalize between 0 and 1
    - Insert val in gaps corresponding to jumps in y (this is the space occupied by the cylinder)
    - Reshape to 201x101x101
    """
    mx = np.max(D)
    mn = np.min(D)
    Dout = (D - mn) / (mx - mn)
    yi = y_rect
    j = 0
    for i in range(NS - 1):
        gap = yi[i + 1] - yi[i]
        if gap > step + 0.01:
            for k in range(1, np.around(gap / step).astype(int)):
                j += 1
                Dout = np.insert(Dout, j, 0, axis=0)
        j += 1
    return Dout.reshape([201, 101, 201])
    
def loss(xcl,ycl,tcl,xb,yb,tb,ub,vb,xtr,ytr,ttr,ctr,weightsCL, \
                                                weightsB,weightsTr,weightsDTr):
    predictB = u_PINN(tf.concat([xb,yb,tb],1))
    predictTrain = u_PINN(tf.concat([xtr,ytr,ttr],1))
    f1,f2,f3,f4,f5 = r_PINN(xcl,ycl,tcl)
    
    #mse_B = tf.reduce_mean(tf.multiply(tf.pow(tf.subtract(predictB[:, 0:1],ub),2)+tf.pow(tf.subtract(predictB[:, 1:2],vb),2),weightsB))
    mse_B = tf.reduce_mean(tf.pow(tf.multiply(tf.subtract(predictB[:, 0:1],ub),weightsB),2)+tf.pow(tf.multiply(tf.subtract(predictB[:, 1:2],vb),weightsB),2))
    mse_ctr = tf.reduce_mean(tf.pow(tf.multiply(tf.subtract(ctr,predictTrain[:, 3:4]),weightsTr),2))
    mse_dtr = tf.reduce_mean(tf.pow(tf.multiply(tf.subtract(tf.subtract(1.0,ctr),predictTrain[:, 4:5]),weightsDTr),2))
    mse_f = tf.reduce_mean(tf.pow(tf.multiply(weightsCL,f1),2) + tf.pow(tf.multiply(weightsCL,f2),2) + tf.pow(tf.multiply(weightsCL,f3),2) + tf.pow(tf.multiply(weightsCL,f4),2)+ tf.pow(tf.multiply(weightsCL,f5),2))
    loss = mse_B + mse_ctr + mse_dtr + mse_f
    return loss, mse_B, mse_ctr, mse_f
    

@tf.function
def grad(model,xcl,ycl,tcl,xb,yb,tb,ub,vb,xtr,ytr,ttr,ctr,weightsCL, \
                                                weightsB,weightsTr,weightsDTr):
    with tf.GradientTape(persistent=True) as tape:
        loss_value, mse_b, mse_trc, mse_f = loss(xcl,ycl,tcl,xb,yb,tb,ub,vb,xtr,ytr,ttr,ctr,weightsCL, \
                                                weightsB,weightsTr,weightsDTr)
        grads = tape.gradient(loss_value,model.trainable_variables)
        gradCL = tape.gradient(loss_value, weightsCL)
        gradB = tape.gradient(loss_value, weightsB)
        gradTr = tape.gradient(loss_value, weightsTr)
        gradDTr = tape.gradient(loss_value, weightsDTr)

    return loss_value, grads, mse_b, mse_trc, mse_f, gradCL, gradB, gradTr, gradDTr

@tf.function
def r_PINN(x,y,t):
    out = u_PINN(tf.concat([x,y,t],1))
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]
    c = out[:, 3:4]
    d = out[:, 4:5]

    u_t = tf.gradients(u,t)[0]
    u_x = tf.gradients(u,x)[0]
    u_y = tf.gradients(u,y)[0]
    L_u = tf.gradients(u_x,x)[0] + tf.gradients(u_y,y)[0]

    v_t = tf.gradients(v,t)[0]
    v_x = tf.gradients(v,x)[0]
    v_y = tf.gradients(v,y)[0]
    L_v = tf.gradients(v_x,x)[0] + tf.gradients(v_y,y)[0]

    p_x = tf.gradients(p,x)[0]
    p_y = tf.gradients(p,y)[0]

    c_t = tf.gradients(c,t)[0]
    c_x = tf.gradients(c,x)[0]
    c_y = tf.gradients(c,y)[0]
    L_c = tf.gradients(c_x,x)[0] + tf.gradients(c_y,y)[0]

    d_t = tf.gradients(d,t)[0]
    d_x = tf.gradients(d,x)[0]
    d_y = tf.gradients(d,y)[0]
    L_d = tf.gradients(d_x,x)[0] + tf.gradients(d_y,y)[0]

    f_1 = u_t + u*u_x + v*u_y + p_x - 0.01*L_u
    f_2 = v_t + u*v_x + v*v_y + p_y - 0.01*L_v
    f_3 = u_x + v_y
    f_4 = c_t + u*c_x + v*c_y - 0.01*L_c
    f_5 = d_t + u*d_x + v*d_y - 0.01*L_d

    return f_1, f_2, f_3, f_4, f_5

    
layer_sizes = [3] + 10*[250] + [5]
u_PINN = neural_net(layer_sizes)

tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.99)
tf_optimizerW = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.99)
weightsCL = tf.Variable(tf.random.uniform([100000, 1]))
weightsB = tf.Variable(tf.random.uniform([20000, 1]))
weightsTr = tf.Variable(tf.random.uniform([100000, 1]))
weightsDTr = tf.Variable(tf.random.uniform([100000, 1]))

weightsCL_List = []
weightsTr_List = []
weightsDTr_List = []
open('stokesOutRain.txt', 'w').close()

for i in range(10):
    weightsCL_List.append(tf.Variable(weightsCL[i*10000:(i+1)*10000,:]))
    weightsTr_List.append(tf.Variable(weightsTr[i*10000:(i+1)*10000,:]))
    weightsDTr_List.append(tf.Variable(weightsDTr[i*10000:(i+1)*10000,:]))

for iter in range(20000):
    #print("iter")
    if (iter == 4000):
        tf_optimizer.learning_rate.assign(0.00080)
    if (iter == 10000):
        tf_optimizer.learning_rate.assign(0.00060)
    if (iter == 12000):
        tf_optimizer.learning_rate.assign(0.00040)
    if (iter == 13000):
        tf_optimizer.learning_rate.assign(0.00020)
    if (iter == 14000):
        tf_optimizer.learning_rate.assign(0.00008)
    if (iter == 16000):
        tf_optimizer.learning_rate.assign(0.00003)
    if (iter == 18000):
        tf_optimizer.learning_rate.assign(0.00001)
    for i in range(10):
        loss_value,grads,b_loss,c_loss,f_loss,gradCL,gradB,gradTr,gradDTr = grad(u_PINN,xcl[i*10000:(i+1)*10000,:],ycl[i*10000:(i+1)*10000,:],tcl[i*10000:(i+1)*10000,:]\
                                                                                 ,xb,yb,tb,ub\
                                                                                 ,vb,xtr[i*10000:(i+1)*10000,:]\
                                                                                 ,ytr[i*10000:(i+1)*10000,:],ttr[i*10000:(i+1)*10000,:],ctr[i*10000:(i+1)*10000,:]\
                                                                                 ,weightsCL_List[i], \
                                                    weightsB,weightsTr_List[i],weightsDTr_List[i])
        tf_optimizer.apply_gradients(zip(grads,u_PINN.trainable_variables))  
        tf_optimizerW.apply_gradients(zip([-gradCL, -gradB, -gradTr, -gradDTr], [weightsCL_List[i], weightsB, weightsTr_List[i], weightsDTr_List[i]]))
    if (iter % 20 == 0):

        #u_PINN_flat = u_PINN(tf.concat([X_flat,Y_flat,T_flat],1))
        U_pred,V_pred,P_pred,C_pred = batch_pred()
        
        pPredErr = P_pred.numpy().T.flatten()
        pErr = P_rect.T.flatten()
        pErr = pErr - (np.mean(pErr))
        pPredErr = pPredErr - (tf.reduce_mean(pPredErr))
        #print(pErr)
        #print(pPredErr)
        
        errU = np.linalg.norm(U_rect.T.flatten()-U_pred.numpy().T.flatten(),2)/np.linalg.norm(U_rect.T.flatten(),2)
        errV = np.linalg.norm(V_rect.T.flatten()-V_pred.numpy().T.flatten(),2)/np.linalg.norm(V_rect.T.flatten(),2)
        errP = np.linalg.norm(pErr-pPredErr,2)/np.linalg.norm(pErr,2)
        errC = np.linalg.norm(C_rect.T.flatten()-C_pred.numpy().T.flatten(),2)/np.linalg.norm(C_rect.T.flatten(),2)

        with open('stokesOutRain.txt', 'a') as f:
            f.write("\n")
            f.write("iter:" + str(iter))
            f.write("\n")
            f.write('U L2 error: %.4e' % (errU))
            f.write("\n")
            f.write('V L2 error: %.4e' % (errV))
            f.write("\n")
            f.write('P L2 error: %.4e' % (errP))
            f.write("\n")
            f.write('C L2 error: %.4e' % (errC))
            f.write("\n")
    
        #print('it. {:-3}: mse_b = {:2.9} mse_trc = {:2.9} mse_f = {:2.9}'.format(iter,b_loss,c_loss,f_loss))
        imgseq_U_pred = get_imgseq(U_pred.numpy())
        imgseq_V_pred = get_imgseq(V_pred.numpy())
        imgseq_P_pred = get_imgseq(P_pred.numpy())
        imgseq_C_pred = get_imgseq(C_pred.numpy())

        imgseq_U = get_imgseq(U_rect)
        imgseq_V = get_imgseq(V_rect)
        imgseq_P = get_imgseq(P_rect)
        imgseq_C = get_imgseq(C_rect)

        #fig, ax = plt.subplots(figsize=(8, 4), nrows=2, ncols=2, dpi=100)
        #ax[0, 0].imshow(imgseq_U[:, :, 100].T, cmap='Spectral')
        #ax[0, 1].imshow(imgseq_V[:, :, 100].T, cmap='Spectral')
        #ax[1, 0].imshow(imgseq_P[:, :, 100].T, cmap='Spectral')
        #ax[1, 1].imshow(imgseq_C[:, :, 100].T, cmap='Spectral')
        #for i in range(2):
        #    for j in range(2):
        #        ax[i, j].add_patch(Circle((50, 50), 11, facecolor='gray'))
        #plt.show()

        fig, ax = plt.subplots(figsize=(8, 4), nrows=2, ncols=2, dpi=100)
        ax[0, 0].imshow(imgseq_U_pred[:, :, 100].T, cmap='Spectral')
        ax[0, 1].imshow(imgseq_V_pred[:, :, 100].T, cmap='Spectral')
        ax[1, 0].imshow(imgseq_P_pred[:, :, 100].T, cmap='Spectral')
        ax[1, 1].imshow(imgseq_C_pred[:, :, 100].T, cmap='Spectral')
        for i in range(2):
            for j in range(2):
                ax[i, j].add_patch(Circle((50, 50), 11, facecolor='gray'))
        #plt.show()
        
        #fig.savefig('./figures/stokesCurrent' + str(iter))
        fig.savefig('./figures/stokesCurrentRain')
        if (iter%200 == 0):
            fig.savefig('./figures/stokesCurrentRain' + str(iter))
        fig.clf()
        plt.close()

    