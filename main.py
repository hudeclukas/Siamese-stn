import tensorflow as tf
import numpy as np
import os
import time

from data_streamer import SUPSIM
from network import siamese_stn


PATH_2_TEXDAT = "D:/Vision_Images/Pexels_textures/TexDat/official"
MODEL_NAME = "siam_stn_001"

MAX_ITERS = 25001
SIAM_MARGIN = 4.5
TRAIN = True

def main():
    supsim = SUPSIM(PATH_2_TEXDAT)
    print("Starting loading data")
    start = time.time() * 1000
    supsim.load_data()
    print((time.time() * 1000) - start, "ms")

    # define network
    siam_stn = siamese_stn(SIAM_MARGIN)
    # define optimizer
    learning_rate = tf.placeholder(tf.float32, shape=[])
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(
        loss=siam_stn.siamese.loss,
        global_step=global_step
    )

    save_dir = 'model/' + MODEL_NAME + '/'
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_ckpt = os.path.join(save_dir, 'checkpoint')


    with tf.Session() as sess:
        if os.path.exists(model_ckpt):
            # restore checkpoint if it exists
            try:
                print("Trying to restore last checkpoint ...")
                last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
                saver.restore(sess, save_path=last_chk_path)
                print("Restored checkpoint from:", last_chk_path)
            except:
                print("Failed to restore checkpoint. Initializing variables instead.")
                sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())

        if TRAIN:

            writer = tf.summary.FileWriter('logs/'+MODEL_NAME+'/', graph=sess.graph)






if __name__ == "__main__":
    main()