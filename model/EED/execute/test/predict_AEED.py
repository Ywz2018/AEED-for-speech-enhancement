import tensorflow as tf
import os
import data_generator
import mir_eval.separation as bss_eval
from mir_eval.eval_measure import calc_pesq, calc_stoi
import numpy as np
import librosa
import config
import soundfile as sf

config.MODE = "test"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def predict(used_model, predicted_SNR, clean_input_path, noise_input_path, noise_model_type, ckpt_id,
            noise_name=None, eval_pesq=True, eval_stoi=True, eval_sdr=True):
    config.NOISE_MODEL_TYPE = noise_model_type
    ckpt_name = "ckpt_" + noise_model_type

    root = os.path.dirname(__file__) + "/../train/"

    ckpt_path = root + "ckpt_set/" + ckpt_name
    # 构造模型
    sess = tf.Session()

    predicted, cost_function, step, memo, test = used_model.model()

    variables_to_restore=tf.trainable_variables()
    used_model.saver = tf.train.Saver(variables_to_restore)

    sess.run(tf.initialize_all_variables())
    if not os.path.exists("ckpt_set"):
        os.mkdir("ckpt_set")
    if os.path.exists(ckpt_path + "/checkpoint"):
        used_model.saver.restore(sess, ckpt_path + "/model.ckpt-" + str(ckpt_id))

    else:
        exit()

    isdr_sum = 0
    osdr_sum = 0
    istoi_sum = 0
    ori_stoi_sum = 0
    ipesq_sum = 0
    ori_pesq_sum = 0
    predict_num = 0

    generator = data_generator.Generator(clean_input_path, noise_input_path)
    for batch in generator.prepare_data_att_PL_simple(snr=predicted_SNR,
                                                      chose_noise=noise_name):  # snr=-5,short_noise=True,noise_in_use=100,start_index=800
        noisy_batch = batch[0]
        noise_batch = batch[1]
        clean_batch = batch[2]
        phase_batch = batch[3]
        refer_voice_batch = batch[4]
        noisy_wave_batch = batch[5]
        voice_lenth_batch = batch[6]

        pred_value, memo_value, test_value = sess.run(fetches=[predicted, memo, test], feed_dict={
            used_model.input_spectrum: noisy_batch,
            used_model.clean_spectrum: clean_batch
        })
        # print("loss:",cost_value_sum)
        print("test", test_value)

        # with open("log/log.txt", "a") as file:
        #     file.write("loss:"+str(cost_value_sum)+"\n")

        for i in range(np.shape(pred_value)[0]):
            print("=" * 80)
            print(str(predict_num + 1))
            print("=" * 80)

            refernce_wav = refer_voice_batch[i][:voice_lenth_batch[i]]
            # 纯净语音相位，临时测试用
            # WINDOW = int(0.032 * 16000)
            # phase = np.transpose(np.angle(librosa.stft(refernce_wav, 512, 256, WINDOW)))

            _pred_spec = pred_value[i] * np.exp(1j * phase_batch[i])  # phase
            predict_wav = librosa.istft(np.transpose(_pred_spec), 256)
            predict_wav = predict_wav[:voice_lenth_batch[i]]

            if eval_stoi:
                pred_stoi = calc_stoi(refernce_wav, predict_wav, 16000)
                ori_stoi = calc_stoi(refernce_wav, noisy_wave_batch[i], 16000)
                stoi_improvement = pred_stoi - ori_stoi
                print("ori_stoi:", ori_stoi)
                print("stoi_improvement:", stoi_improvement)
                istoi_sum += stoi_improvement
                ori_stoi_sum += ori_stoi
                with open("/home/ywz/person/语音增强/att_PL/log/temp_measure_mini_note", "a", encoding="utf8") as f:
                    f.write(" istoi:" + str(stoi_improvement) + "o_sdr:" + str(ori_stoi) + "\n")
            if eval_pesq:
                pred_pesq = calc_pesq(refernce_wav, predict_wav, 16000)
                ori_pesq = calc_pesq(refernce_wav, noisy_wave_batch[i], 16000)
                pesq_improvement = pred_pesq - ori_pesq
                print("ori_pesq:", ori_pesq)
                print("pesq_improvement:", pesq_improvement)
                ipesq_sum += pesq_improvement
                ori_pesq_sum += ori_pesq
                with open("/home/ywz/person/语音增强/att_PL/log/temp_measure_mini_note", "a", encoding="utf8") as f:
                    f.write(" ipesq:" + str(pesq_improvement) + "opesq:" + str(ori_pesq) + "\n")
            if eval_sdr:
                sdr, sir, sar, _ = bss_eval.bss_eval_sources(refernce_wav, predict_wav)
                ori_sdr, ori_sir, ori_sar, _ = bss_eval.bss_eval_sources(refernce_wav, noisy_wave_batch[i])
                isdr = sdr - ori_sdr
                print("o_sdr:", ori_sdr)
                print("isdr:", isdr)
                isdr_sum += isdr
                osdr_sum += ori_sdr
                with open("/home/ywz/person/语音增强/att_PL/log/temp_measure_mini_note", "a", encoding="utf8") as f:
                    f.write(" isdr:" + str(isdr) + "o_sdr:" + str(ori_sdr) + "\n")
            predict_num += 1

            # if predict_num >= 4:
            #     break
            if not os.path.exists("wav"):
                os.mkdir("wav")
                os.mkdir("wav/ori")
                os.mkdir("wav/pred")
                os.mkdir("wav/ref")
            if predict_num % 3 == 0:
                librosa.output.write_wav(
                    "wav/ori/" + str(predict_num) + ".wav", noisy_wave_batch[i],
                    16000)
                sf.write("wav/pred/" + str(predict_num) + ".wav",
                         predict_wav, 16000)
                sf.write("wav/ref/" + str(predict_num) + ".wav",
                         refernce_wav, 16000)

        if predict_num >= 100:
            break

    with open("/home/ywz/person/语音增强/att_PL/log/temp_measure_note", "a", encoding="utf8") as f:
        f.write("=" * 80 + "\n")
        f.write(ckpt_name + " " + str(ckpt_id) + "\n")
        f.write("=" * 80 + "\n")

    if eval_stoi:
        gistoi = istoi_sum / predict_num
        gostoi = ori_stoi_sum / predict_num
        gstoi = gistoi + gostoi
        print("gistoi", gistoi)
        print("gostoi", gostoi)

        with open("/home/ywz/person/语音增强/att_PL/log/temp_measure_note", "a", encoding="utf8") as f:
            f.write("gistoi:" + str(gistoi) + " gostoi:" + str(gostoi) + " gstoi:" + str(gstoi) + "\n")
    if eval_pesq:
        gipesq = ipesq_sum / predict_num
        gopesq = ori_pesq_sum / predict_num
        gpesq = gipesq + gopesq
        print("gipesq", gipesq)
        print("gopesq", gopesq)

        with open("/home/ywz/person/语音增强/att_PL/log/temp_measure_note", "a", encoding="utf8") as f:
            f.write("gipesq:" + str(gipesq) + " gopesq:" + str(gopesq) + " gpesq:" + str(gpesq) + "\n")
    if eval_sdr:
        gisdr = isdr_sum / predict_num
        gosdr = osdr_sum / predict_num
        gsdr = gisdr + gosdr
        print("gisdr:", gisdr)
        print("gosdr", gosdr)

        with open("/home/ywz/person/语音增强/att_PL/log/temp_measure_note", "a", encoding="utf8") as f:
            f.write("gisdr:" + str(gisdr) + " gosdr:" + str(gosdr) + " gsdr:" + str(gsdr) + "\n")


if __name__ == "__main__":
    clean_input_path = config.TEST_CLEAN_INPUT_PATH
    noise_input_path = config.TEST_NOISE_INPUT_PATH
    noise_model_type = config.NOISE_MODEL_TYPE

    ckpt_id = 60
    from model.EED.structure.direct import model_AEED_for_test as model

    used_model = model.AEED()

    with tf.Session() as sess:
        predict(used_model, predicted_SNR=-5, clean_input_path=clean_input_path, noise_input_path=noise_input_path,
                noise_model_type=noise_model_type, noise_name=None, ckpt_id=ckpt_id)
