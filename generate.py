# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:17:43 2018

@author: Ethan
"""
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.client import timeline

from CausalConv_muC import conv_vae

def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return 0

def main():
    n = int(sys.argv[1])
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        #logdir = './save_dir'
        logdir = './save'

        # Create network.
        net = conv_vae(sess)

        # Set up session
       # sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        # Saver for storing checkpoints of the model.
        variables_to_restore = [var for var in tf.trainable_variables()
                        if (var.name.startswith('g')
                        or var.name.startswith('d'))
                        and "e2_convf" not in var.name]
        #print(variables_to_restore)
        #input()
        saver = tf.train.Saver(var_list=variables_to_restore, max_to_keep=5)
        try:
            saved_global_step = load(saver, sess, logdir)
            # if is_overwritten_training or saved_global_step is None:
            #     # The first training step will be saved_global_step + 1,
            #    # therefore we put -1 here for new or overwritten trainings.
            #   saved_global_step = -1
        except:
           print("Something went wrong while restoring checkpoint. "
                      "We will terminate training to avoid accidentally overwriting "
                      "the previous model.")
           raise
        net.generate_select(n)


if __name__ == '__main__':
    main()
