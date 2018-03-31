import tensorflow as tf
import numpy as np
import os
import time

from data_streamer import SUPSIM
from network import siamese_stn, Similarity


PATH_2_TEXDAT = "D:/Vision_Images/Pexels_textures/TexDat/official"
MODEL_NAME = "siam_stn_001"

MAX_ITERS = 25001
BATCH_SIZE = 24
SIAM_MARGIN = 4.5
TRAIN = True

def main():
    supsim = SUPSIM(PATH_2_TEXDAT)
    print("Starting loading data...")
    start = time.time() * 1000
    supsim.load_data()
    print((time.time() * 1000) - start, "ms")

    # define network
    siam_stn = siamese_stn(SIAM_MARGIN, BATCH_SIZE)
    # define similarity operations
    sim_ops = Similarity()
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

            writer = tf.summary.FileWriter('logs/' + MODEL_NAME + '/', graph=sess.graph)
            merge_summary = tf.summary.merge_all()
            for epoch in range(0, 10):
                print("Epoch {:01d}".format(epoch))

                for step in range(0, MAX_ITERS):
                    step = MAX_ITERS * epoch + step
                    (batch_1, batch_2, labels) = supsim.train.next_teacher_batch(batch_size=BATCH_SIZE)
                    l_rate = 0.002 / (1.75 * float(epoch + 1))
                    board_summary,_, loss_v = sess.run(
                        [merge_summary, train_step, siam_stn.siamese.loss], feed_dict={
                            siam_stn.net_1_input: batch_1,
                            siam_stn.net_2_input: batch_2,
                            siam_stn.siamese.y: labels,
                            siam_stn.siamese.dropout_keep_prob: 0.6,
                            learning_rate: l_rate
                        })
                    if step < 50 or step % 10 == 0:
                        print("Step: [{:04d}.] --> loss: |{:3.8f}|...".format(step, loss_v))

                    if step < 50 or step % 100 == 0 :
                        writer.add_summary(board_summary, step)

                    if step % 5000 == 0 and epoch > 1:
                        print("Teacher is validating train...")

                        sim_all = None
                        lbl_all = None
                        idx_all = None
                        for i in range(4):
                            x_1, x_2, x_l, idx = supsim.train.next_validation_batch(50)
                            vec1 = siam_stn.siamese.network_1.eval({siam_stn.net_1_input: x_1})
                            vec2 = siam_stn.siamese.network_2.eval({siam_stn.net_2_input: x_2})
                            similarity = sess.run(sim_ops.euclid, {sim_ops.vec1: vec1, sim_ops.vec2: vec2})
                            result = list(zip(similarity, x_l))
                            print(result)
                            if not type(sim_all) == np.ndarray:
                                sim_all = np.array(similarity)
                            else:
                                sim_all = np.concatenate((sim_all, similarity))
                            if not type(lbl_all) == np.ndarray:
                                lbl_all = x_l
                            else:
                                lbl_all = np.concatenate((lbl_all, x_l))
                            if not type(idx_all) == np.ndarray:
                                idx_all = idx
                            else:
                                idx_all = np.concatenate((idx_all, idx))
                        # Verify well learned classes and delete them from teacher
                        tp = [i[0] for i, l, s in zip(idx_all, lbl_all, sim_all) if l == 1. and s <= 0.8]
                        tn = [i for i, l, s in zip(idx_all, lbl_all, sim_all) if l == 0. and s >= 1.5]
                        tp_u, tp_c = np.unique(tp, return_counts=True)
                        for u, c in zip(tp_u, tp_c):
                            if c > 15:
                                if u in supsim.train.teacher.fn:
                                    supsim.train.teacher.fn.remove(u)
                        tn_u, tn_c = np.unique(tn, return_counts=True, axis=0)
                        for u, c in zip(tn_u, tn_c):
                            if c > 10:
                                if u in supsim.train.teacher.fp:
                                    supsim.train.teacher.fp.remove(u)
                        # Select classes that had high number of FP or FN and add them to teacher
                        fn = [i[0] for i, l, s in zip(idx_all, lbl_all, sim_all) if l == 1. and s > 1.2]
                        fp = [i for i, l, s in zip(idx_all, lbl_all, sim_all) if l == 0. and s < 0.65]
                        fn_u, fn_c = np.unique(fn, return_counts=True)
                        supsim.train.teacher.fn += [i for i, c in zip(fn_u, fn_c) if c > 4]
                        if len(fp) > 0:
                            fp = np.sort(fp, axis=1)
                            fp_u = np.unique(fp, axis=0)
                            fp_u_s, fp_s_c = np.unique(fp, return_counts=True)
                            fp_i = [i for i, c in zip(fp_u_s, fp_s_c) if c > 3]
                            if len(fp_i) > 0:
                                fp_r = [u for i in fp_i for u in fp_u if u[0] == i or u[1] == i]
                                supsim.train.teacher.fp += np.unique(fp_r, axis=0).tolist()

                    if step % 10000 == 0 and step > 0:
                        save_path = saver.save(sess, model_ckpt, step)
                        print("Model saved to file %s" % save_path)


if __name__ == "__main__":
    main()