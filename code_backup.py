from __future__ import division
import theano
import theano.tensor as T
import theano.tensor.signal.conv
import numpy as np
import cv2
from tqdm import tqdm
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from keras.utils import np_utils, generic_utils
from sklearn.cluster import KMeans


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(floatX(p.get_value() * 0.))
        v = theano.shared(floatX(p.get_value() * 0.))
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

def draw_gray(la,g_val):
    img = np.zeros(la.shape)
    for i in range(n_clusters):
        img = img + (la == i)*g_val[i]
    
    return img

#################
## Build loss function and the updates function
## version 1: use if with theano
#################
c = 1;
delta = 1;
lr = 1;
## build up the loss function
params = []
for i in range(n_clusters):
    grayv = theano.shared(np.asarray(g_ini[i], dtype=theano.config.floatX))
    params.append(grayv)

loss_contrast = 0.
w_sum = 0.
for i in range(n_clusters-1):
    for j in range(i+1,n_clusters):
        l_delta = l_ini[i] - l_ini[j]
        a_delta = a_ini[i] - a_ini[j]
        b_delta = b_ini[i] - b_ini[j]
        s_delta = T.sqrt(T.sqr(l_delta) + T.sqr(a_delta) + T.sqr(b_delta))
        
        flag = 1 * ((l_delta > 0) and (a_delta >0) and (b_delta>0)) + (-1)*((l_delta < 0) and (a_delta < 0) and (b_delta < 0))   
        g_delta = params[i]-params[j]
#         if flag==0:
#             loss_t = T.abs_(g_delta) - c * s_delta
#         else:
#             loss_t = g_delta - c * flag * s_delta              
        if flag==0:
            g_delta = T.log((T.abs_(params[i]-params[j]))/delta)/np.log(256/delta) * 255
            loss_t = g_delta - c * s_delta
        else:
            g_delta = T.log(flag*(params[i]-params[j])/delta)/np.log(256/delta) * 255
            loss_t = g_delta - c * s_delta
        w_ij = n_pixels[i] * n_pixels[j]
        w_sum = w_sum + w_ij
        loss_contrast = loss_contrast + w_ij * T.sqr(loss_t)

loss_contrast = loss_contrast / w_sum
## 
## still need a boundary term
##
g_update = Adam(loss_contrast,params,lr)  
f = theano.function([],outputs = loss_contrast,updates=g_update)



#################
## Build loss function and the updates function
## version 2: use theano scan with shared variable of the grayv
#################
## for computing loss of one step
# def onepari(grayv_i, l_i, a_i, b_i, grayv, l_ini, a_ini, b_ini):
#     l_delta = l_i - l_ini
#     a_delta = a_i - a_ini
#     b_delta = b_i - b_ini
#     s_delta = T.sqrt(T.sqr(l_delta) + T.sqr(a_delta) + T.sqr(b_delta))

#     flag = 1 * T.and_(T.and_(T.gt(l_delta, 0),T.gt(a_delta,0)),T.gt(b_delta,0)) + (-1)*T.and_(T.and_(T.lt(l_delta, 0),T.lt(a_delta,0)),T.lt(b_delta,0))  
#     g_delta = grayv_i-grayv
#         if flag==0:
#             loss_t = T.abs_(g_delta) - c * s_delta
#         else:
#             loss_t = g_delta - c * flag * s_delta             
    g_delta_nl = T.switch(T.eq(flag, 0), T.log((T.abs_(g_delta)+1)/delta)/np.log(256/delta) * 255, T.log((flag*g_delta + 1)/delta)/np.log(256/delta) * 255)
    loss_t = g_delta_nl - c * s_delta
    loss_c = T.sum(T.sqr(loss_t)) 
#     if flag==0:
#         g_delta = T.log((T.abs_(g_delta))/delta)/np.log(256/delta) * 255
#         loss_t = g_delta - c * s_delta
#     else:
#         g_delta = T.log(flag*g_delta/delta)/np.log(256/delta) * 255
#         loss_t = g_delta - c * s_delta
#     loss_c = T.sqr(loss_t)
    return loss_c

## build up the loss function
def buildloss(c=1,delta=1):

    grayv = theano.shared(np.asarray(g_ini, dtype=theano.config.floatX))
    l_ini = T.fvector('l_ini')
    a_ini = T.fvector('a_ini')
    b_ini = T.fvector('b_ini')
    lr = T.fscalar('lr')

    loss_c, updates = theano.scan(fn=onepari,
                                  outputs_info=None,
                                  sequences=[grayv, l_ini, a_ini, b_ini],
                                  non_sequences=[grayv, l_ini, a_ini, b_ini])
    loss_contrast = T.sum(loss_c)

    g_update = Adam(loss_contrast,[grayv],lr)  
    f = theano.function([l_ini,a_ini,b_ini,lr],outputs = loss_contrast,updates=g_update)
    return f, grayv

#################
## Build loss function and the updates function
## version 3: use theano scan with separate adam class
#################
## for computing loss of one step
def onepari(grayv_i, l_i, a_i, b_i, grayv, l_ini, a_ini, b_ini,c,delta):
    l_delta = l_i - l_ini
    a_delta = a_i - a_ini
    b_delta = b_i - b_ini
    s_delta = T.sqrt(T.sqr(l_delta) + T.sqr(a_delta) + T.sqr(b_delta))

    flag = 1 * T.and_(T.and_(T.gt(l_delta, 0),T.gt(a_delta,0)),T.gt(b_delta,0)) + (-1)*T.and_(T.and_(T.lt(l_delta, 0),T.lt(a_delta,0)),T.lt(b_delta,0))  
    g_delta = grayv_i-grayv
#         if flag==0:
#             loss_t = T.abs_(g_delta) - c * s_delta
#         else:
#             loss_t = g_delta - c * flag * s_delta             
    g_delta_nl = T.switch(T.eq(flag, 0), T.log((T.abs_(g_delta)+1)/delta)/np.log(256/delta) * 255, T.log((flag*g_delta + 1)/delta)/np.log(256/delta) * 255)
    loss_t = g_delta_nl - c * s_delta
    loss_c = T.sum(T.sqr(loss_t)) 
#     if flag==0:
#         g_delta = T.log((T.abs_(g_delta))/delta)/np.log(256/delta) * 255
#         loss_t = g_delta - c * s_delta
#     else:
#         g_delta = T.log(flag*g_delta/delta)/np.log(256/delta) * 255
#         loss_t = g_delta - c * s_delta
#     loss_c = T.sqr(loss_t)
    return loss_c

## build up the loss function
def buildloss(c=1,delta=1):
    grayv = T.fvector('grayv')
    l_ini = T.fvector('l_ini')
    a_ini = T.fvector('a_ini')
    b_ini = T.fvector('b_ini')
#     lr = T.fscalar('lr')

    loss_c, updates = theano.scan(fn=onepari,
                                  outputs_info=None,
                                  sequences=[grayv, l_ini, a_ini, b_ini],
                                  non_sequences=[grayv, l_ini, a_ini, b_ini,c,delta])
    loss_contrast = T.sum(loss_c)/2
    grads = T.grad(loss_contrast, [grayv])
    outputs = []
    outputs.append(loss_contrast)
    outputs = outputs + grads
    
#     g_update = Adam(loss_contrast,[grayv],lr)  
    f = theano.function([grayv,l_ini,a_ini,b_ini],outputs=outputs)
    return f

class Adamupdate():
    def __init__(self,params):
        self.i = 0
        self.m=[]
        self.v=[]

        for p in params:
            m = p * 0.
            self.m.append(m)
            v = p * 0.
            self.v.append(v)
    
    def update(self,params,grads,lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        params_n = []
        i_t = self.i + 1
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (np.sqrt(fix2) / fix1)
        for idx in range(len(params)):
            p = params[idx]
            g = grads[idx]
            m = self.m[idx]
            v = self.v[idx]

            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * np.square(g)) + ((1. - b2) * v)
            g_t = m_t / (np.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            
            params[idx] = p_t
            self.m[idx] = m_t
            self.v[idx] = v_t
        
        self.i = self.i + 1


#########################
## Main script
##########################
#### load the images
path_images = '/data/bjin/MyDecolor/images/'
path_results = '/data/bjin/MyDecolor/results/'

img_name = '1.png'
img = cv2.imread(path_images + img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_JJia = cv2.imread(path_results + '2012_JJia/' + img_name)
lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
lab_img[:,:,0] = lab_img[:,:,0]/255*100
l_img,a_img,b_img = cv2.split(lab_img.astype(np.float32))
# l_img = l_img/255*100
sz = img.shape

print sz

#### cluster the colors
n_clusters = 50 # heuristicly set the number
clt = KMeans(n_clusters = n_clusters)
img_d2 = lab_img.reshape([sz[0]*sz[1],3])
clt.fit(img_d2)
la = clt.labels_
la = la.reshape([sz[0],sz[1]])

fig = plt.figure(figsize=(20, 4))
ax = fig.add_subplot(131)
ax.imshow(img)
ax.set_title('ori')
ax.axis('off')

ax = fig.add_subplot(132)
ax.imshow(la,cmap='gray')
ax.set_title('clusters')
ax.axis('off')

# build the statistics 
n_pixels = np.zeros((n_clusters,)).astype(np.float32)
l_ini = np.zeros((n_clusters,)).astype(np.float32)
a_ini = np.zeros((n_clusters,)).astype(np.float32)
b_ini = np.zeros((n_clusters,)).astype(np.float32)
for i in range(n_clusters):
    n_pixels[i] = np.sum(la==i)
    l_ini[i] = np.sum(l_img * (la == i))/n_pixels[i]
    a_ini[i] = np.sum(a_img * (la == i))/n_pixels[i]
    b_ini[i] = np.sum(b_img * (la == i))/n_pixels[i]
g_ini = np.copy(l_ini)
img_ini = draw_gray(la,g_ini)

ax = fig.add_subplot(133)
ax.imshow(img_ini,cmap='gray')
ax.set_title('initial')
ax.axis('off')

plt.show()

##### compute the grayv
print grayv.get_value()

n_iter = 2000
progbar = generic_utils.Progbar(n_iter)
for i in range(n_iter):
    loss_c = f(l_ini,a_ini,b_ini,lr=0.1)
    progbar.add(1., values=[("train loss", loss_c),])

print grayv.get_value()

##### visualize the results
fig = plt.figure(figsize=(20, 4))
ax = fig.add_subplot(141)
ax.imshow(img)
ax.set_title('ori')
ax.axis('off')

ax = fig.add_subplot(142)
ax.imshow(l_img,cmap='gray')
ax.set_title('l channel')
ax.axis('off')

ax = fig.add_subplot(143)
img_g = draw_gray(la,grayv)
im = ax.imshow(img_g,cmap='gray')
ax.set_title('result_vlog')
ax.axis('off')

ax = fig.add_subplot(144)
ax.imshow(img_JJia,cmap='gray')
ax.set_title('result_JJia')
ax.axis('off')
plt.show()



############
### Local term on the grey image, each pixel's grey value is a  variable.
###########
gimg = theano.shared(np.asarray(gini, dtype=theano.config.floatX))
## local term
k = np.array([[1,-1]]).astype(np.float32)
gtemp = T.reshape(gimg,sc)
l_delta = theano.tensor.signal.conv.conv2d(l_img,k, border_mode='valid')
a_delta = theano.tensor.signal.conv.conv2d(a_img,k, border_mode='valid')
b_delta = theano.tensor.signal.conv.conv2d(b_img,k, border_mode='valid')
s_delta = T.sqrt(T.sqr(l_delta) + T.sqr(a_delta) + T.sqr(b_delta))
g_delta_h = T.abs_(theano.tensor.signal.conv.conv2d(gtemp,k,border_mode='valid'))
loss_f1 = T.sum(T.sqr(g_delta_h - c * s_delta))

k = np.array([[1],[-1]]).astype(np.float32)
gtemp = T.reshape(gimg,sc)
l_delta = theano.tensor.signal.conv.conv2d(l_img,k, border_mode='valid')
a_delta = theano.tensor.signal.conv.conv2d(a_img,k, border_mode='valid')
b_delta = theano.tensor.signal.conv.conv2d(b_img,k, border_mode='valid')
s_delta = T.sqrt(T.sqr(l_delta) + T.sqr(a_delta) + T.sqr(b_delta))
g_delta_v = T.abs_(theano.tensor.signal.conv.conv2d(gtemp,k,border_mode='valid'))
loss_f2 = T.sum(T.sqr(g_delta_v - c * s_delta))

loss_TV = T.sum(g_delta_h) + T.sum(g_delta_v)

gimgt = gimg.flatten()
## global term
for i in range(N):
    idx = np.random.permutation(lvec)
    gtemp = gimgt[idx]
    g_delta = T.abs_(gimgt - gtemp)
    
    l_delta = l_img_v - l_img_v[idx]
    a_delta = a_img_v - a_img_v[idx]
    b_delta = b_img_v - b_img_v[idx]
    s_delta = np.sqrt(np.square(l_delta) + np.square(a_delta) + np.square(b_delta))
    
    loss_f3 = T.sum(T.sqr(g_delta - c * s_delta))

# loss_boundary = T.sum(alpha * T.gt(gimg,255) + alpha * T.lt(gimg, 0))
# loss_boundary = 
    
loss_delta =loss_f3 + alpha*loss_TV 
g_update = Adam(loss_delta,[gimg],lr)    
f = theano.function([],outputs=[loss_delta, loss_TV],updates=g_update)





############
### version 4, 2017, January 16.
### The global approach, change the flag function
###########
###################
#### required functions
###################
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def draw_gray(la,g_val,mask=1):
    img = np.zeros(la.shape)
    n_clusters = np.max(la) + 1
    for i in range(n_clusters):
        img = img + (la == i)*g_val[i]
    
    imgn = cv2.GaussianBlur(img,(7,7),0) * (1 - mask) + img * mask
    return imgn

def draw_color(la,imgrgb,mask=1):
    mask = np.expand_dims(mask,axis=2)
    mask = np.repeat(mask,3,axis=2)
    r_img,g_img,b_img = cv2.split(imgrgb)
    n_clusters = np.max(la) + 1
    for i in range(n_clusters):
        r_img[la==i] = np.mean(r_img[la==i])
        g_img[la==i] = np.mean(g_img[la==i])
        b_img[la==i] = np.mean(b_img[la==i])
    
    imgn = np.stack((r_img,g_img,b_img),axis=2).astype(np.float32)
    imgn = cv2.GaussianBlur(imgn,(7,7),0) * (1 - mask) + imgn * mask
    return imgn

## for computing loss of one step
def onepari(grayv_i, l_i, a_i, b_i, R_i, G_i, B_i, grayv, l, a, b, R, G, B):
    l_delta = l_i - l
    a_delta = a_i - a
    b_delta = b_i - b
    s_delta = T.sqrt(T.sqr(l_delta) + T.sqr(a_delta) + T.sqr(b_delta))
    
    R_delta = R_i - R
    G_delta = G_i - G
    B_delta = B_i - B
     
    flag = 1 * T.and_(T.and_(T.gt(R_delta, 0),T.gt(G_delta,0)),T.gt(B_delta,0)) + (-1)*T.and_(T.and_(T.lt(R_delta, 0),T.lt(G_delta,0)),T.lt(B_delta,0))  

#     g_delta = (T.log(grayv_i/delta + 1) - T.log(grayv/delta + 1))/np.log(255/delta+1)*255
    g_delta = grayv_i - grayv
    g_delta_nl = T.switch(T.eq(flag, 0), T.abs_(g_delta), flag*g_delta)
    loss_c =T.sum(T.sqr(g_delta_nl - s_delta))
#     w_delta = n_i * n_pixels / w_all
#     loss_c = T.sum(w_i * T.abs_(loss_t)) 
#     loss_c = T.sum(T.abs_(loss_t)) 
    return loss_c

## build up the loss function
def buildloss():
    w = []
    for i in range(9):
        if i < 3:
            w.append(theano.shared(floatX(1/3)))
        else:
            w.append(theano.shared(floatX(0)))
    lr = theano.shared(floatX(0.01))
    
    R = T.fvector('R')
    G = T.fvector('G')
    B = T.fvector('B')
    l = T.fvector('l')
    a = T.fvector('a')
    b = T.fvector('b')

    grayv = w[0] * R + w[1] * G + w[2]*B + w[3] *R*G + w[4]*G*B + w[5]*B*R + w[6]*(T.sqr(R)) + w[7]*(T.sqr(G)) +w[8]*(T.sqr(B)) 
    
    loss_c, updates = theano.scan(fn=onepari,
                                  outputs_info=None,
                                  sequences=[grayv, l, a, b, R, G, B],
                                  non_sequences=[grayv, l, a, b, R, G, B])
    
    loss_contrast = T.sum(loss_c)/2
    w_update = Adam(loss_contrast,w,learning_rate=lr)
#     grads = T.grad(loss_contrast, w)
    outputs = []
    outputs.append(loss_contrast)
#     outputs = outputs + grads
    
    f = theano.function([R,G,B,l,a,b],outputs=outputs,updates=w_update)
    return f,w,lr

class Adamupdate():
    def __init__(self,params):
        self.i = 0
        self.m=[]
        self.v=[]

        for p in params:
            m = p * 0.
            self.m.append(m)
            v = p * 0.
            self.v.append(v)
    
    def update(self,params,grads,lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        params_n = []
        i_t = self.i + 1
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (np.sqrt(fix2) / fix1)
        for idx in range(len(params)):
            p = params[idx]
            g = grads[idx]
            m = self.m[idx]
            v = self.v[idx]

            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * np.square(g)) + ((1. - b2) * v)
            g_t = m_t / (np.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            
            params[idx] = p_t
            self.m[idx] = m_t
            self.v[idx] = v_t
        
        self.i = self.i + 1
        
def Adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates
        
## Build up the theano function and the matlab session
f,w,lr = buildloss()
matlab = matlab_wrapper.MatlabSession()

def color2gray(path_images, img_name):
    ## load the images
    img = cv2.imread(path_images + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    R_img = img[:,:,0].astype(np.float32) / 255
    G_img = img[:,:,1].astype(np.float32) / 255
    B_img = img[:,:,2].astype(np.float32) /255

    l_img = (lab_img[:,:,0]).astype(np.float32) / 255 
    a_img = (lab_img[:,:,1]).astype(np.float32) / 255
    b_img = (lab_img[:,:,2]).astype(np.float32) / 255

    sz = img.shape
    n_p = sz[0] * sz[1]
    print sz, n_p
    
    ## automatically determine the numner of clusters
    matlab.put('filename', path_images + img_name)
    matlab.put('cdist', 30)
    matlab.put('minsize', 20)
    matlab.eval('findkmeans')
    numklabels = matlab.get('numklabels')
    klabels = matlab.get('klabels')
    numclabels = matlab.get('numclabels')
    clabels = matlab.get('clabels')
    print 'number of clusters: ', numklabels
    mask = np.zeros(clabels.shape)
    for i in range(numclabels):
        n_p_conn = np.sum(clabels==i)    
        mask[clabels==i] = n_p_conn > 0.001*n_p

    ## build the statistics 
    n_pixels = np.zeros((numklabels,)).astype(np.float32)
    R_ini = np.zeros((numklabels,)).astype(np.float32)
    G_ini = np.zeros((numklabels,)).astype(np.float32)
    B_ini = np.zeros((numklabels,)).astype(np.float32)
    l_ini = np.zeros((numklabels,)).astype(np.float32)
    a_ini = np.zeros((numklabels,)).astype(np.float32)
    b_ini = np.zeros((numklabels,)).astype(np.float32)

    for i in range(numklabels):
        n_pixels[i] = np.sum(klabels==i)
        R_ini[i] = np.sum(R_img * (klabels == i))/n_pixels[i]
        G_ini[i] = np.sum(G_img * (klabels == i))/n_pixels[i]
        B_ini[i] = np.sum(B_img * (klabels == i))/n_pixels[i]

        l_ini[i] = np.sum(l_img * (klabels == i))/n_pixels[i]
        a_ini[i] = np.sum(a_img * (klabels == i))/n_pixels[i]
        b_ini[i] = np.sum(b_img * (klabels == i))/n_pixels[i]
    # w_all = ((n_pixels.sum()**2 - n_pixels.dot(n_pixels)) / 2).astype(np.float32)

    ## compute the adjacency matrix between colors
    # w_ij = np.zeros((numklabels,numklabels)).astype(np.float32)
    # for i in range(sz[0]-1):
    #     for j in range(sz[1]-1):
    #         w_ij[klabels[i,j],klabels[i+1,j]] = w_ij[klabels[i,j],klabels[i+1,j]] + 1
    #         w_ij[klabels[i,j],klabels[i,j+1]] = w_ij[klabels[i,j],klabels[i,j+1]] + 1
    # np.fill_diagonal(w_ij,0)
    # w_ij = w_ij + w_ij.T 
    # w_ij = (w_ij.T / np.sum(w_ij,axis=1)).T
    
    ## draw the clustering results
#     fig = plt.figure(figsize=(20, 4))
#     ax = fig.add_subplot(141)
#     ax.imshow(img)
#     ax.set_title('ori')
#     ax.axis('off')

#     ax = fig.add_subplot(142)
#     ax.imshow(klabels,cmap='gray')
#     ax.set_title('clusters')
#     ax.axis('off')
    
#     g_ini = np.copy(l_ini)
#     imgg_ini = draw_gray(klabels,g_ini,mask)

#     ax = fig.add_subplot(143)
#     ax.imshow(imgg_ini,cmap='gray')
#     ax.set_title('initial')
#     ax.axis('off')

#     imgc_ini = draw_color(klabels,img,mask)
#     ax = fig.add_subplot(144)
#     ax.imshow(imgc_ini.astype(np.uint8),vmin=0,vmax=255)
#     ax.set_title('initial')
#     ax.axis('off')

#     plt.show()
    
    ## perform the decolorization optimization
    #### initialize the parameters and the learning rate
    for i in range(9):
        if i < 3:
            w[i].set_value(floatX(1/3))
        else:
            w[i].set_value(floatX(0))
    wv = [i.get_value() for i in w]
    print wv

    lr.set_value(100)
    print lr.get_value()
    
    ## perform the optimization
    n_iter = 200
    progbar = generic_utils.Progbar(n_iter)
    for i in range(n_iter):
        loss = f(R_ini,G_ini,B_ini,l_ini,a_ini,b_ini)
        progbar.add(1., values=[("train loss", loss[0]),])
    
    wv = [i.get_value() for i in w]
    print wv

    ## draw the decolorization results
    img_g =  wv[0] * R_img + wv[1] * G_img + wv[2]*B_img + wv[3] *R_img*G_img + wv[4]*G_img*B_img + wv[5]*B_img*R_img + wv[6]*(R_img**2) + wv[7]*(G_img**2) +wv[8]*(B_img**2)
    
    return img_g

## process all images in the folder
test_idx = '1'
path_base = '/data/bjin/MyDecolor/dataset/Cadik/'
path_images = path_base + 'images/'
path_results_mine = path_base + 'mine/test' + test_idx + '/'
path_results_2012Lu = path_base + '2012_Lu/'
path_results_2013Song = path_base + '2013_Song/'
path_results_2015Du = path_base + '2015_Du/'
path_results_2015Liu = path_base + '2015_Liu/'

if not os.path.isdir(path_results_mine):
    os.mkdir(path_results_mine)
files = os.listdir(path_images)
n_files = len(files)

for img_name in files:
    try: 
        img_g = color2gray(path_images, img_name)
    except:
        matlab = matlab_wrapper.MatlabSession()
        img_g = color2gray(path_images, img_name)
    img_g = (img_g - np.min(img_g))/(np.max(img_g) - np.min(img_g))
    
    ## plot the respective results
    fig = plt.figure(figsize=(20, 4))
    
    ax = fig.add_subplot(161)
    img = cv2.imread(path_images + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title('ori')
    ax.axis('off')

    ax = fig.add_subplot(162)
    img_2012Lu = cv2.imread(path_results_2012Lu + img_name)
    ax.imshow(img_2012Lu,cmap='gray')
    ax.set_title('2012 Lu')
    ax.axis('off')

    ax = fig.add_subplot(163)
    img_2013Song = cv2.imread(path_results_2013Song + img_name)
    ax.imshow(img_2013Song,cmap='gray')
    ax.set_title('2013 Song')
    ax.axis('off')

    ax = fig.add_subplot(164)
    img_2015Du = cv2.imread(path_results_2015Du + img_name)
    ax.imshow(img_2015Du,cmap='gray')
    ax.set_title('2015 Du')
    ax.axis('off')
    
    ax = fig.add_subplot(165)
    img_2015Liu = cv2.imread(path_results_2015Liu + img_name)
    ax.imshow(img_2015Liu,cmap='gray')
    ax.set_title('2015 Liu')
    ax.axis('off')
    
    ax = fig.add_subplot(166)
    ax.imshow(img_g,cmap='gray')
    ax.set_title('mine')
    ax.axis('off')
    
#     cv2.imwrite(path_results_mine+img_name,(img_g*255).astype(np.uint8))

## evaluation 
matlab = matlab_wrapper.MatlabSession()

matlab.put('thrNum', 15)
matlab.put('imNum', 24)
matlab.put('path_base', path_base)
matlab.eval('Computemetrics')
CCPR = matlab.get('CCPR')
CCFR = matlab.get('CCFR')
Escore = matlab.get('Escore')
print CCPR



