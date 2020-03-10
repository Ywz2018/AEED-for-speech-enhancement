import tensorflow as tf
from model.EED.structure.direct import model_AEED as model
import os
import data_generator
import config
import numpy as np

type = "c_Noisex92_AEED mdB"
config.NOISE_MODEL_TYPE = str(type)
config.MODE = "train"

clean_input_path = config.TRAIN_CLEAN_INPUT_PATH
noise_input_path = config.TRAIN_NOISE_INPUT_PATH

valid_clean_input_path = config.VALID_CLEAN_INPUT_PATH
valid_noise_input_path = config.VALID_NOISE_INPUT_PATH

# 构造模型
AEED_model = model.AEED()
predicted, cost_function, step, memo, test = AEED_model.model()

AEED_model.saver = tf.train.Saver(max_to_keep=15)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.initialize_all_variables())
    if not os.path.exists("ckpt_set"):
        os.mkdir("ckpt_set")
    if os.path.exists("ckpt_set/ckpt_" + config.NOISE_MODEL_TYPE + "/checkpoint"):
        AEED_model.saver.restore(sess, "ckpt_set/ckpt_" + config.NOISE_MODEL_TYPE + "/model.ckpt")
        if not os.path.exists("log"):
            os.mkdir("log")
        with open("log/" + type + "log.txt", "a") as file:
            file.write("\n" + "=" * 100 + "\n" + "=" * 100 + "\n")
    else:
        if not os.path.exists("log"):
            os.mkdir("log")
        with open("log/" + type + "log.txt", "w") as file:
            file.write("")

    epoch = 80

    for i in range(1, 1 + epoch):
        print("=" * 80)
        print(str(i))
        print("=" * 80)

        generator = data_generator.Generator(clean_input_path, noise_input_path)
        j = 0
        cost_sum = 0
        with open("log/" + type + "log.txt", "a") as file:
            file.write("epoch:" + str(i) + "\n")
        for batch in generator.prepare_data_att_PL_simple(snr=-5):  # ,short_noise=True,noise_in_use=800
            noisy_batch = batch[0]
            noise_batch = batch[1]
            clean_batch = batch[2]
            pred_LPS_value, cost_value, _, memo_value, test_result = sess.run(
                fetches=[predicted, cost_function, step, memo, test], feed_dict={
                    AEED_model.input_spectrum: noisy_batch,
                    AEED_model.noise_spectrum: noise_batch,
                    AEED_model.clean_spectrum: clean_batch
                })
            batch_size = np.shape(batch[0])[0]
            print("loss:", cost_value / batch_size)
            # print("test:",test_result)
            # print("LPS:",pred_LPS_value)
            j += 1
            cost_sum += cost_value / batch_size

            with open("log/" + type + "log.txt", "a") as file:
                file.write("loss:" + str(cost_value / batch_size) + "\n")

        # 验证
        valid_loss_sum = 0
        k = 0
        generator = data_generator.Generator(valid_clean_input_path, valid_noise_input_path)
        for batch in generator.prepare_data_att_PL_tiny(short_noise=True):
            noisy_batch = batch[0]
            noise_batch = batch[1]
            clean_batch = batch[2]
            pred_LPS_value, cost_value, memo_value, test_result = sess.run(
                fetches=[predicted, cost_function, memo, test], feed_dict={
                    AEED_model.input_spectrum: noisy_batch,
                    AEED_model.noise_spectrum: noise_batch,
                    AEED_model.clean_spectrum: clean_batch
                })
            batch_size = np.shape(batch[0])[0]
            print("valid_loss:", cost_value / batch_size)
            k += 1
            with open("log/" + type + "log.txt", "a") as file:
                file.write("valid_loss:" + str(cost_value / batch_size) + "\n")
            valid_loss_sum += cost_value / batch_size

        if i % 5 == 0 and i > 0:
            AEED_model.saver.save(sess, "ckpt_set/ckpt_" + config.NOISE_MODEL_TYPE + "/model.ckpt", i)

        print("epoch mean loss:", cost_sum / j)
        with open("log/" + type + "log.txt", "a") as file:
            file.write("epoch mean loss:" + str(cost_sum / j) + "\n")
        print("epoch valid loss:", cost_sum / k)
        with open("log/" + type + "log.txt", "a") as file:
            file.write("epoch valid loss:" + str(valid_loss_sum / k) + "\n")
