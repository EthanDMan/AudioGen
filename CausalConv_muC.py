# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 16:57:34 2018

@author: Ethan
"""

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from tensorflow_utils import *
from sig_utils import *
from losses import *

from AudioLoader import AudioLoader

import sys

def spec_to_sig(Y):
    #Y = spec_log(Y)
    Y = tf.transpose(Y, perm=[0,2,1,3])
    rY, iY = tf.split(Y, 2, axis=3)
    C = tf.squeeze(tf.complex(rY, iY),axis = [3])
    #CC = tf.reverse(tf.conj(C), axis = [1])
    #F = tf.concat([C, CC], axis = 1)
    #F = tf.transpose(F, [0,2,1])
    #F = tf.cast(F, tf.complex64)
    y = tf.contrib.signal.inverse_stft(C,512,100)
    #y = mu_decode(y)
    return y[:,0:25600]

def sig_to_spec(x):
    #x = mu_encode(x)
    S = tf.contrib.signal.stft(x,512,100,pad_end=True)
    y = tf.stack([tf.real(S), tf.imag(S)], axis=-1)
    y = tf.transpose(y, perm=[0,2,1,3])
    #y = log_spec(y)
    return y

def quantize(x):
    max_x = tf.reduce_max(tf.abs(x),[1,2,3],True)
    xx = x/max_x
    y = tf.round(xx*2**31)
    return y


class conv_vae():
    
   def  __init__(self,sess):
       self.sess = sess
       
       self.loader = AudioLoader()
       
       save_dir = 'checkpoints/'
       if not os.path.exists(save_dir):
           os.makedirs(save_dir)
           
       self.save_path = os.path.join(save_dir, 'model')
       
       self.build_model()
       
       self.init = tf.global_variables_initializer()
       self.sess.run(self.init)
       
   def build_model(self):
       self.inputs_ = tf.placeholder(tf.float32, (None, 25600), name='inputs')
       self.specs = (sig_to_spec(self.inputs_))
       #self.z = tf.placeholder(tf.float32, (None,32), name='z')
       self.d = tf.placeholder(tf.float32,(None,len(self.loader.class_names())))
       self.h = tf.placeholder(tf.float32,(None,138))
       self.p = tf.placeholder(tf.float32,(None,128))
       
       self.gen_ops = []
       self._dis_train_ops = []
       
        # discriminator loss
        
       (self.encoded, self.z_mean, self.z_log_sigma)  = self.encoder()
       #self.encoded = self.encoder()
       self.z = self.sample()
       #self.decoded = tf.reverse(self.decoder(),[1])
       self.decoded = self.decoder()
       #self.decoded = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.decoded)
       
       d_logit_real, self.class_vec_real = self.discriminator(self.specs)
       d_logit_fake, self.class_vec_fake = self.discriminator(self.decoded, is_reuse=True)
       
       self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
       self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_real)))
       
       classification_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.class_vec_real, labels=self.d))
       classification_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.class_vec_fake, labels=self.d))
       self.d_loss = (self.d_loss_real + self.d_loss_fake+1e-8) + (classification_loss_real+1e-8)
       
       d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
       g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')
       
       dis_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.d_loss, var_list=d_vars)
       dis_ops = [dis_op] + self._dis_train_ops
       self.dis_optim = tf.group(*dis_ops)
       
       gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
       #gan_loss += classification_loss_fake #- self.d_loss_fake
       
       log_specs = log_spec(self.specs)
       log_decoded = log_spec(self.decoded)
       
     
       orig = self.inputs_
       recon = spec_to_sig(self.decoded)
       
       m_orig = mu_encode(orig)
       m_recon = mu_encode(recon)
       
       s_orig = spec_to_sig(log_specs)
       s_recon = spec_to_sig(log_decoded)
       
       t_orig = spec_to_sig(tf.tanh(self.specs))
       t_recon = spec_to_sig(tf.tanh(self.decoded))
       
       t_log_orig = spec_to_sig(tf.tanh(log_specs))
       t_log_recon = spec_to_sig(tf.tanh(log_decoded))
       
       Eo = energy_spec(self.specs)
       #Er = energy_spec(self.decoded)
       mse = tf.reduce_mean((tf.reduce_sum(tf.square((self.specs)-
                                                    (self.decoded))/Eo,[1,2,3])))
       #mse = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.square(log_specs)-
       #                                             tf.square(log_decoded))/Eo,[1,2,3])))
       mse2 = tf.reduce_mean(tf.reduce_mean(tf.square(tf.square(log_spec((self.specs)))-
                                                    tf.square(log_spec((self.decoded))))/tf.log(Eo),[1,2,3]))

       #self.gen_cost_f = f_loss2(t_recon,t_orig) + f_loss2(m_recon,m_orig) + f_loss2(s_recon,s_orig)
       #self.gen_cost_f += f_loss2(recon,orig) + f_loss2(t_log_recon,t_log_orig)
       self.gen_cost_f = f_corr(recon,orig) #+ f_corr(s_recon,s_orig) + f_corr(t_recon,t_orig) 
       #self.gen_cost_f += f_corr(m_recon,m_orig) #+ f_corr(s_recon,s_orig) + f_corr(t_log_recon,t_log_orig) 
       
       #self.gen_cost += (f_loss(t_recon,t_orig)) + f_loss(mu_encode(recon),mu_encode(orig))
       #self.gen_cost = (f_loss(s_recon,s_orig)) #+ f_loss(recon,orig)
       #self.gen_cost += f_loss2(recon,orig)
       #self.gen_cost += f_constraint2(recon,orig) #+ f_constraint2(s_recon,s_orig) + f_constraint2(mu_encode(recon),mu_encode(orig)))
       #self.gen_cost += f_constraint2(recon,orig) + f_constraint2(s_recon,s_orig) 
       
       #self.gen_cost_t =  ((t_loss2(t_log_recon,t_log_orig))) + t_loss2(mu_encode(recon),mu_encode(orig))
       #self.gen_cost_t +=  ((t_loss2(recon,orig)) + t_loss2(s_recon,s_orig)) + ((t_loss2(t_recon,t_orig)))
       self.gen_cost_t =  (t_corr(recon,orig)) #+ (t_corr(m_recon,m_orig)) #+ (t_corr(t_log_recon,t_log_orig)))/10
       #self.gen_cost_t +=  t_corr(s_recon,s_orig) + ((t_corr(t_recon,t_orig)))
       #self.gen_cost += t_constraint(orig,recon)
       
       self.gen_cost_s = s_corrt(self.decoded, self.specs) #+s_corrt(log_decoded, log_specs) /10)
       #self.gen_cost_s += 50*(s_corrf(self.decoded, self.specs) + s_corrf(log_decoded, log_specs)/10)
       #self.gen_cost_s += 10*s_corr(self.decoded,self.specs)
       
       self.gen_cost = self.gen_cost_f + self.gen_cost_t #+ mse/200
       #self.gen_cost = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.decoded, self.specs),[1,2,3]))/(256**2*2)
      # self.gen_cost += 10000*tf.reduce_mean(tf.square(tf.reduce_mean(tf.square(orig),-1)-
      #                                 tf.reduce_mean(tf.square(recon),-1)))
       #self.gen_cost += tf.reduce_mean(tf.reduce_sum(tf.abs(self.decoded),[1,2,3]))/100000
       #self.gen_cost += 1e-7*tf.reduce_mean(tf.reduce_sum(tf.abs(tf.reduce_sum(tf.square(self.specs),[2])-
       #                                  tf.reduce_sum(tf.square(self.decoded),[2])),[1,2]))
       #self.gen_cost += 10*tf.reduce_mean(tf.abs(tf.reduce_sum(tf.square(orig),-1)-
       #                                  tf.reduce_sum(tf.square(recon),-1)))
       #self.gen_cost += (gan_loss)/100
       #self.gen_cost += fft_loss(recon,orig)
       #self.gen_cost += 10*(t_loss(mu_encode(orig),mu_encode(recon)) + t_loss((orig),(recon))) 
       #self.gen_cost +=  (f_constraint(recon,orig)) #+ f_constraint(mu_encode(orig),mu_encode(recon)))
       #self.gen_cost += 10*t_constraint(orig,recon) #+ t_constraint(mu_encode(orig),mu_encode(recon)))
       #self.gen_cost += -f_loss_c/10
       #self.gen_cost += 1e5*regularizer(recon, orig)
       #self.gen_cost += 1000*regularizer2(recon, orig)
       
       opt = [tf.train.AdamOptimizer(0.01).minimize(self.gen_cost, var_list=g_vars)] + self.gen_ops 
       self.opt = tf.group(*opt) 
       
   def encoder(self, name='g_'):
       with tf.variable_scope(name):
        ### Encoder
            S = (self.specs[:,:256,:,:])#,'specs', _ops=self.gen_ops)
            #noise = tf.random_normal(tf.shape(self.inputs_), mean = 0.0, stddev = 0.5, dtype = tf.float32) 
            e1_convlt  = causal_conv2(S, 4, k_h=1, k_w=256, name='e1_convst')
            e1_convst  = causal_conv2(S, 12, k_h=3, k_w=16, name='e1_convlt')
            #e1_convlf = conv2d(self.inputs_, 4, k_h=8, k_w=2, name='e1_conlf')
            #e1_convgf = conv2d(self.inputs_, 4, k_h=16, k_w=1, name='e1_congf', dilate=4)
            e1_conv =  lrelu(batch_norm(tf.concat([e1_convlt, e1_convst], axis=3, name='e1_conv'),
                                       name='e0_batchnorm', _ops=self.gen_ops))
            e2_convf  = causal_conv2(e1_conv, 6, k_h=1, k_w=128, name='e2_convf')
            e2_convt  = causal_conv2(e1_conv, 18, k_h=3, k_w=16, name='e2_convt')
            e2_conv =  lrelu(batch_norm(tf.concat([e2_convf, e2_convt], axis=3, name='e2_conv'),
                                       name='e1_batchnorm', _ops=self.gen_ops))
            e3_convf  = causal_conv2(e2_conv, 8, k_h=1, k_w=64, name='e3_convf')
            e3_convt  = causal_conv2(e2_conv, 24, k_h=3, k_w=16, name='e3_convt')
            e3_conv =  lrelu(batch_norm(tf.concat([e3_convf, e3_convt], axis=3, name='e3_conv'),
                                       name='e2_batchnorm', _ops=self.gen_ops))
            e4_convf  = causal_conv2(e3_conv, 10, k_h=1, k_w=32, name='e4_convf')
            e4_convt  = causal_conv2(e3_conv, 30, k_h=3, k_w=16, name='e4_convt')
            e4_conv =  lrelu(batch_norm(tf.concat([e4_convf, e4_convt], axis=3, name='e4_conv'),
                                       name='e3_batchnorm', _ops=self.gen_ops))
            encoded = lrelu(batch_norm(conv2d(e4_conv, 2, k_h=3, k_w=3, d_h=1, d_w=1, name='encoded'),
                                  name='e4_batchnorm', _ops=self.gen_ops))
            #e = dense(tf.reshape(encoded,[tf.shape(self.inputs_)[0],2048]), 2048,2048+self.d.shape[1],name='z_mean')
            e = (tf.reshape(encoded,[tf.shape(self.inputs_)[0],16*16*2]))
            z_mean = dense(e, 512,32,name='z_mean')
            z_log_sigma = dense(e,512,32,name='z_log_sigma')
            return e, z_mean, z_log_sigma
    
   def decoder(self, name="g_"):
       with tf.variable_scope(name):
        ### Decoder
        #sample = tf.concat([self.encoded, self.d], axis = 1, name = 'sample')
        cond = tf.concat([self.d, self.h], axis = 1, name = 'conditions')
        #ccc = tanh(dense(cond, cond.shape[1], 128,name='ccc'))
        
        #sample_c = tf.concat([self.z, ccc], axis = 1, name = 'sample2')
        sample_c = tf.concat([self.encoded, self.d], axis = 1, name = 'sample2')
    
        projection = (dense(sample_c,sample_c.shape[1],16*16*2,name='projection',identity=True))
        #p = tf.concat([projection, self.p], axis = 1, name = 'p')
        #proj = dense(p, p.shape[1], 16*16*2, name='proj')
        p_spec = tf.reshape(projection, [tf.shape(self.inputs_)[0],16,16,2])
        #projection = tf.reshape(dense(self.z,32,2048,name='projection'),[tf.shape(self.inputs_)[0],32,32,2])
        
        d0_convf  = causal_deconv2(p_spec, 10, k_h=1, k_w=32, name='d0_convf')
        d0_convt  = causal_deconv2(p_spec, 30, k_h=3, k_w=16, name='d0_convt')
        
        d0_conv =  lrelu(batch_norm(tf.concat([d0_convf, d0_convt], axis=3, name='d0_conv'),
                               name='d0_batchnorm', _ops=self.gen_ops))
        #d0_mask = sigmoid(gate_causal_conv(d0_conv,1,h=self.p,k_h=32,name='d0_gate'))
        #d0_drop = tf.nn.dropout(d0_conv,0.95,name='d0_drop')
        
        d1_convf  = causal_deconv2(d0_conv, 8, k_h=1, k_w=64, name='d1_convf')
        d1_convt  = causal_deconv2(d0_conv, 24, k_h=3, k_w=16, name='d1_convt')
        d1_conv =  lrelu(batch_norm(tf.concat([d1_convf, d1_convt], axis=3, name='d1_conv'),
                               name='d1_batchnorm', _ops=self.gen_ops))
        #d1_mask = sigmoid(gate_causal_conv(d1_conv,1,h=self.p,k_h=64,name='d1_gate'))
        #d1_drop = tf.nn.dropout(d1_conv,0.9,name='d1_drop')
        
        d2_convf  = causal_deconv2(d1_conv, 6, k_h=1, k_w=128, name='d2_convf')
        d2_convt  = causal_deconv2(d1_conv, 18, k_h=3, k_w=16, name='d2_convt')
        d2_conv =  lrelu(batch_norm(tf.concat([d2_convf, d2_convt], axis=3, name='d2_conv'),
                               name='d2_batchnorm', _ops=self.gen_ops))
        #d2_mask = sigmoid(gate_causal_conv(d2_conv,1,h=self.p,k_h=128,name='d2_gate'))
        #d2_drop = tf.nn.dropout(d2_conv,0.85,name='d2_drop')
        
        d3_convf  = causal_deconv2(d2_conv, 4, k_h=1, k_w=256, name='d3_convf')
        d3_convt  = causal_deconv2(d2_conv, 12, k_h=3, k_w=16, name='d3_convt')
        
        d3_conv =  lrelu(batch_norm(tf.concat([d3_convf, d3_convt], axis=3, name='d3_conv'),
                               name='d3_batchnorm', _ops=self.gen_ops))
        #d3_drop = tf.nn.dropout(d3_conv,0.8,name='d3_drop')
        
        
        d4_conv = lrelu(causal_conv(d3_conv,8,k_h=1,k_w=256,d_h=1,d_w=1,name='d4_conv'))
        v = sigmoid(tf.reshape(dense(self.p,self.p.shape[1],256,name='gate'),[-1,256,1,1]),name='gate2')
        #v = sigmoid(tf.reshape(dense(self.p,self.p.shape[1],256*2,name='gate3'),[-1,256,1,2]),name='gate4')
        #d4_gate = sigmoid(gate_causal_conv(d3_conv,1,h=self.p,k_h=256,name='d4_gate'))
        #d4_drop = tf.nn.dropout(d4_conv,0.75,name='d4_drop')
        
        #d4_sum = d4_conv
        
        
        #decoded = (causal_conv(d4_sum,4, k_h=1,k_w=256, d_h=1, d_w=1,name='test'))
        #decoded_2 = causal_conv(decoded,4, k_h=8,k_w=16, d_h=1, d_w=1,name='test2')
        yy = (conv2d(d4_conv,2, k_h=1,k_w=256, d_h=1, d_w=1,name='output'))
        #yyy = convf2d(yy,256,name='out2')
        #decoded = deconv2d(d3_conv,2)
        #Y = tf.pad(decoded_2, [[0, 0], [0, 1], [0, 0], [0, 0]])
        Y = tf.pad(yy, [[0, 0], [0, 1], [0, 0], [0, 0]])
        return Y
    
   def sample(self):
        batch_n = tf.shape(self.inputs_)[0]
        dim = 32
        epsilon = tf.random_normal(tf.stack([batch_n, dim])) 
        z  = self.z_mean + tf.multiply(epsilon, tf.exp(self.z_log_sigma))
        return z
    
   def discriminator(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()
            # 512 -> 256
            h0_convf  = causal_conv(data[:,:256,:,:], 4, k_h=1, k_w=128, name='h0_convf')
            h0_convt  = causal_conv(data[:,:256,:,:], 12, k_h=3, k_w=16, name='h0_convt')
            h0_conv =  lrelu(batch_norm(tf.concat([h0_convf, h0_convt], axis=3, name='h0_conv'),
                               name='h0_batchnorm', _ops=self._dis_train_ops))

            # 256 -> 128
            h1_convf  = causal_conv(h0_conv, 4, k_h=1, k_w=64, name='h1_convf')
            h1_convt  = causal_conv(h0_conv, 12, k_h=3, k_w=16, name='h1_convt')
            h1_conv =  lrelu(batch_norm(tf.concat([h1_convf, h1_convt], axis=3, name='h1_conv'),
                               name='h1_batchnorm', _ops=self._dis_train_ops))
            
            # 128 -> 64
            h2_convf  = causal_conv(h1_conv, 4, k_h=1, k_w=64, name='h2_convf')
            h2_convt  = causal_conv(h1_conv, 12, k_h=3, k_w=16, name='h2_convt')
            h2_conv =  lrelu(batch_norm(tf.concat([h2_convf, h2_convt], axis=3, name='h2_conv'),
                               name='h2_batchnorm', _ops=self._dis_train_ops))
            
            h3_conv2d = relu(causal_conv(h2_conv, 2, k_h=3, k_w=3, d_h=1, d_w=1,  name='h3_conv2d'))
            h3_batchnorm = batch_norm(h3_conv2d, name='h3_batchnorm', _ops=self._dis_train_ops)
            
            h4_dense = tf.reshape(h3_batchnorm,[tf.shape(self.inputs_)[0],32*32*2], name='h4_dense')
            
            logits = dense(h4_dense, 32*32*2, 512, name='logits')
            
            h5_dense = dense(logits, 512, 64, name='h5_dense')
            prediction = dense(h5_dense, 64,1)
            class_vec = (dense(h4_dense, 32*32*2, 11, name='classes'))

            return (logits, class_vec)
        
    
   def load(self):
       saver = tf.train.import_meta_graph('checkpoints/model.meta')
       saver.restore(self.sess, tf.train.latest_checkpoint('checkpoints/'))
       print("Model loaded.")
        

   def train(self):       
        batch_size = 5
        #for e in range(epochs):
        batch, instruments, cond = next(self.loader.load_next(batch_size))
        h = cond[:,128:]
        p = cond[:,:128]
        batch_cost, _ = self.sess.run([self.gen_cost, self.opt], feed_dict={self.inputs_: batch,
                                                         self.d : instruments,
                                                         self.p : p,
                                                         self.h : h})
        d_loss, _ = self.sess.run([self.d_loss,self.dis_optim], feed_dict={self.inputs_: batch,
                                                         self.d : instruments,
                                                         self.p : p,
                                                         self.h : h})
        batch_cost, _ = self.sess.run([self.gen_cost, self.opt], feed_dict={self.inputs_: batch,
                                                         self.d : instruments,
                                                         self.p : p,
                                                         self.h : h})
        print("Gen loss: {:.4f}".format(batch_cost), "Disc loss: {:.4f}".format(d_loss))
        
            
   def generate_rand(self,test=False):
        (x, i, c) = self.loader.load_test()
        Y = self.sess.run(self.decoded, feed_dict={self.inputs_: x,
                                                   self.d : i,
                                                   self.p : c[:,:128],
                                                   self.h : c[:,128:]})
        ys = spec_to_sig(tf.convert_to_tensor(Y))
        y, yf = self.sess.run([ys, normalise_spectra(tf.abs(tf.fft(tf.cast(ys,tf.complex64))))])
        XS = sig_to_spec(tf.convert_to_tensor(x,dtype=tf.float32))
        X = self.sess.run(XS)
        #YY = self.sess.run(log_spec(tf.convert_to_tensor(Y,dtype=tf.float32)))
        xf = self.sess.run(normalise_spectra(tf.abs(tf.fft(tf.cast((tf.convert_to_tensor(x)),tf.complex64)))))
        x = self.sess.run(spec_to_sig(XS))
        self.loader.save({'data': x, 'recon': y, 'spec' : X, 'spec_r':Y})
        plt.figure()
        plt.plot(yf)
        plt.plot(xf)
        temp = np.zeros((256,256,3))
        temp[:,:,0:2] = Y[0,:256,:,:]
        plt.figure()
        plt.imshow(temp)
        plt.figure()
        temp[:,:,0:2] = X[0,:256,:,:]
        plt.imshow(temp)
        plt.figure()
        plt.plot(y[0,:])
        plt.plot(x[0,:])        
        plt.show()
        
        
   def generate_select(self, n=2):
        while True:
            print(self.loader.class_names())
            option = input()
            if option.lower() in "quit":
                break
            try:
                (x, i,c) = next(self.loader.select_by_sample(option, n))
                Y = self.sess.run(self.decoded, feed_dict={self.inputs_: x,
                                                   self.d : i,
                                                   self.p : c[:,:128],
                                                   self.h : c[:,128:]})
                y = self.sess.run(spec_to_sig(tf.tanh(tf.convert_to_tensor(Y,dtype=tf.float32))))
                X = self.sess.run(sig_to_spec(tf.convert_to_tensor(x,dtype=tf.float32)))
                temp = np.zeros((256,256,3))
                for i in range(n):
                    self.loader.save({'data': x[i,:], 'recon': y[i,:], 'spec' : X[i,:,:,:], 'spec_r':Y[i,:,:,:]})
                    plt.figure()
                    plt.plot(y[i,:])
                    plt.plot(x[i,:])
                    temp[:,:,0:2] = Y[i,:256,:,:]
                    plt.figure()
                    plt.imshow(temp)
                    plt.figure()
                    temp[:,:,0:2] = X[i,:256,:,:]
                    plt.imshow(temp)
                    plt.show()
            except:
                print('what?')
                
def main():
    args = sys.argv[1]
    epochs = int(sys.argv[2])
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
            AE = conv_vae(sess)
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)
            if args == "train":
                AE.train(epochs)
            else:
                AE.load()
            AE.generate_rand(True)
            AE.generate_select()    

if __name__ == "__main__": main()