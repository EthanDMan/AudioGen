# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:41:37 2018

@author: Ethan
"""

import numpy as np
import tensorflow as tf

from tensorflow_utils import *

def log_spec(S):
    sig = tf.sign(S)
    LS = sig*tf.log(1.0+tf.abs(S))/np.log(10.0)
    return LS

def spec_log(LS):
    sig = tf.sign(LS)
    S = sig*(tf.exp(tf.abs(LS))-1.0)
    return S

def mu_spec(x):
    xx = x/(1e-20+tf.reduce_max(tf.abs(x),[2],True))
    return mu_encode(xx)

def mu_encode(x):
    x = normalise(x)
    N = 2**32-1
    return tf.sign(x)*tf.log(1+N*tf.abs(x))/np.log(1+N)
        
def mu_decode(x):
    N = 2**32-1
    return tf.sign(x)*((1.0/N)*(((1+N)**tf.abs(x))-1))

def KL_divergence(p, q):
    p = tf.stop_gradient(p)
    """ Compute KL divergence of two vectors, K(p || q)."""
    kl = tf.reduce_sum(p*tf.log(p+1e-10),axis=-1) - tf.reduce_sum(p*tf.log(q+1e-10),axis=-1)
    #Z = q/(p+1e-12)
    #kl = tf.reduce_mean(-tf.reduce_sum(p*tf.log(tf.clip_by_value(Z,1e-10,1e10)),axis=-1))
    return tf.reduce_mean(kl)

def KL_divergence2(p, q):
    p = tf.stop_gradient(p)
    """ Compute KL divergence of two vectors, K(p || q)."""
    Z = p/(q+1e-10)
    #Z = tf.clip_by_value(p,1e-5,1e10)/(tf.clip_by_value(q,1e-5,1e10))
    #Z = tf.where(tf.is_nan(Z), 1e6*tf.ones_like(Z), Z)
    kl = tf.reduce_mean(tf.reduce_sum(((p)*tf.log(tf.clip_by_value(Z,1e-10,1e10))),axis=-1))
    return (kl)
    
def KL_divergence3(p, q):
    p = tf.stop_gradient(p)
    """ Compute KL divergence of two vectors, K(p || q)."""
    Z = p/(q+1e-10)
    kl = tf.reduce_mean(tf.reduce_sum((p)*tf.log(tf.clip_by_value(Z,1e-10,1e10)),axis=-1))
    return (kl)
    
def normalise_spec(s):
    s = tf.abs(s)
    return s/(tf.reduce_sum(s,axis=[1,2,3],keepdims=True)+1e-10)

def normalise_spec2(s):
    return s/(tf.reduce_sum(tf.abs(s),axis=1,keepdims=True)+1e-10)

def JSD(p, q):
    M = (p+q)/2
    JS = KL_divergence(p, M) + KL_divergence(q, M)
    return JS

def JSD2(p, q):
    p = tf.abs(normalise_spec(p))
    q = tf.abs(normalise_spec(q))
    M = (p+q)/2
    JS = KL_divergence2(p, M) + KL_divergence2(q, M)
    return JS

def JSD3(p, q):
    p2 = p/tf.clip_by_value((tf.reduce_sum(tf.abs(p),axis=1,keepdims=True)),1e-8,1e10)
    q2 = q/tf.clip_by_value((tf.reduce_sum(tf.abs(q),axis=1,keepdims=True)),1e-8,1e10)
    M2 = (p2+q2)/2
    JS = KL_divergence(p2, M2) + KL_divergence(q2, M2)
    return JS/2

def compress(f):
    maximum = tf.reduce_max(f, axis=-1, keepdims=True)*0.1
    condition = tf.greater_equal(f, maximum)
    f = tf.where(condition,f,tf.zeros_like(f))
    return f
    
def fft_loss(x,y):
    rx, ix = fft(x)
    ry, iy = fft(y)
    
    prx, pry = relu(rx), relu(ry)
    nrx, nry = relu(-rx), relu(-ry)
    pix, piy = relu(ix), relu(iy)
    nix, niy = relu(-ix), relu(-iy)
    
    #reg = tf.reduce_sum(tf.abs(prx-pry),axis=-1) + tf.reduce_sum(tf.abs(nrx-nry),axis=-1)
    #reg += tf.reduce_sum(tf.abs(pix-piy),axis=-1) + tf.reduce_sum(tf.abs(nix-niy),axis=-1)
    #reg = tf.reduce_mean(tf.sqrt(reg))/100
    
    prx, pry = (normalise_spectra((prx))) , normalise_spectra(pry)
    nrx, nry = (normalise_spectra((nrx))) , normalise_spectra(nry)
    pix, piy = (normalise_spectra((pix))) , normalise_spectra(piy)
    nix, niy = (normalise_spectra((nix))) , normalise_spectra(niy)
      
    #x_pdfs = tf.concat()
    loss =   KL_divergence(prx,pry) #JSD(prx, pry)
    loss +=  KL_divergence(nrx,nry) #JSD(nrx, nry)
    loss +=  KL_divergence(pix,piy) #JSD(pix, piy)
    loss +=  KL_divergence(nix,niy) #JSD(nix, niy)
    
    lpr = tf.log(pry/(1-pry)+1e-8)
    lnr = tf.log(nry/(1-nry)+1e-8)
    lpi = tf.log(piy/(1-piy)+1e-8)
    lni = tf.log(niy/(1-niy)+1e-8)
    
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=prx,logits=lpr))
    loss2 += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=nrx,logits=lnr))
    loss2 += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=pix,logits=lpi))
    loss2 += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=nix,logits=lni))
    
    return  loss/4
   
def CSD(x,y):
    fx = fft(x)
    fy = fft(y)
    
    fxr = normalise_spectra(tf.real(fx))
    fxi = normalise_spectra(tf.imag(fx))
    fyr = normalise_spectra(tf.real(fy))
    fyi = normalise_spectra(tf.imag(fy))
    
def spec_loss(p,q):
    loss2 = KL_divergence3(relu(normalise_spec2(q)), relu(normalise_spec2(p)))
    loss2 += KL_divergence3(relu(normalise_spec2(-q)), relu(normalise_spec2(-p)))
    return loss2/2

def spec_loss2(p,q):
    loss = JSD3(relu(p),relu(q))
    loss += JSD3(relu(-1*p),relu(-1*q))
    return loss/2
    
def fft(x):
    f = tf.fft((tf.cast(x, tf.complex64)))[:,:12800]
    return tf.real(f), tf.imag(f)

def fft_tanh(x):
    f = tf.atanh(tf.fft((tf.cast(x, tf.complex64))))
    return tf.real(f), tf.imag(f)

def normalise_spectra_log(f): 
    sig = tf.sign(f)
    f= tf.log1p(tf.abs(f))
    return sig*f 

def normalise_spectra(f):
    f = f[:,0:int(25600/2)]
    #f = tf.clip_by_value(f, -1, 5)+1
    f = f/(tf.reduce_sum(tf.abs(f),axis=-1,keepdims=True)+1e-10)
    f = tf.squeeze(f) 
    return f    
    
def normalise(x): 
    x = x - tf.reduce_mean(x,-1,keepdims=True)
    x = x / tf.reduce_max(tf.abs(x),-1,keepdims=True)
    return x

def rms_normalise(x): 
    x = x - tf.reduce_mean(x,-1,keepdims=True)
    x = x / tf.sqrt(tf.reduce_mean(tf.square(x),-1,keepdims=True))
    return x

def spec_normalise(x): 
    #x = x - tf.reduce_mean(x,-1,keepdims=True)
    E = energy_spec(x)
    x = x / E
    return x

def energy_spec(x):
    E = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(x),-1,True)),[1,2],True)
    return E

    
 