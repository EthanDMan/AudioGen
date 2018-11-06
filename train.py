"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
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

BATCH_SIZE = 20
CHECKPOINT_EVERY = 50
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3



def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


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
    epochs = int(sys.argv[1])
    if len(sys.argv) > 2:
        option = (sys.argv[2])
    else:
        option = 'new'
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        #logdir = './save_dir'
        logdir = './save_dir'
        # Even if we restored the model, we will treat it as new training
        # if the trained model is written into an arbitrary location.

        # Create coordinator.
        coord = tf.train.Coordinator()

        # Create network.
        net = conv_vae(sess)

        # Set up session
       # sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        # Saver for storing checkpoints of the model.
        variables_to_restore = [var for var in tf.trainable_variables()
                        if (var.name.startswith('g')
                        or var.name.startswith('d'))
                        and 'v/' not in var.name]
        #print(variables_to_restore)
        #input()
        saver = tf.train.Saver(var_list=variables_to_restore, max_to_keep=5)
        if option == 'load':
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
        else:
            saved_global_step = 0

        #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        step = 1
        last_saved_step = saved_global_step
        try:
            for step in range(1,epochs):
                start_time = time.time()
                print("Epoch: {}/{}...".format(step, epochs))
                net.train()
                
                #if step % 20 == 0:
                    #save(saver, sess, logdir, step)
                    #last_saved_step = step

        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()
        finally:
            step += last_saved_step
            #save(saver, sess, logdir, step)
            #coord.request_stop()
            #coord.join(threads)
            net.generate_rand(True)
            net.generate_select()
            if input('save?') == 'y':
                saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=5)
                save(saver, sess, logdir, saved_global_step+1)


if __name__ == '__main__':
    main()
