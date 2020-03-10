import tensorflow as tf
import numpy as np
import config


class AEED():
    def __init__(self):
        self.batch_size = None
        self.LSTM_weight_size = 784
        self.embedding_size = 40
        self.target_spectra_size = 257
        self.sample_rate = 512
        self.layer_num_to_each_target = 3  # 两个encoder中LSTM的层数
        self.frame_size = 235
        self.use_FC = False  # False:RC-AEED, True:FC-AEED
        self.life_long_mem = tf.get_variable(name="SpkLifeLongMemory", shape=(1, self.embedding_size),
                                             dtype=tf.float32,
                                             initializer=tf.zeros_initializer())

    def embedding_array_update(self, mix_embedding_array, memo_vector):
        # 将长期记忆向量复制扩展成与带噪embedding矩阵相同的维度 (embedding_size)->(batch_size,frame_size,spectra_size,embedding_size)
        memo_vector = tf.expand_dims(memo_vector, 0)
        memo_vector = tf.expand_dims(memo_vector, 0)
        memo_vector = tf.expand_dims(memo_vector, 0)
        memo_vector = tf.concat([memo_vector for _ in range(32)], 0)
        memo_vector = tf.concat([memo_vector for _ in range(self.frame_size)], 1)
        memo_vector = tf.concat([memo_vector for _ in range(self.target_spectra_size)], 2)

        # mask (batch_size, frame_size, spectra_size)
        mask = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(mix_embedding_array, memo_vector), axis=3))
        # 复制重叠mask为stacked_mask (batch_size, frame_size, spectra_size)->(batch_size, frame_size, spectra_size,embedding_size)
        stacked_mask = tf.stack([mask] * self.embedding_size, axis=3)
        # 对带噪embedding矩阵进行更新（点乘stacked_mask）
        enhanced_mix_embedding_array = tf.multiply(stacked_mask, mix_embedding_array)
        return enhanced_mix_embedding_array, mask

    def model(self):
        self.input_spectrum = tf.placeholder(dtype=tf.float32,
                                             shape=(self.batch_size, self.frame_size, self.target_spectra_size))
        self.clean_spectrum = tf.placeholder(dtype=tf.float32,
                                             shape=(self.batch_size, self.frame_size, self.target_spectra_size))
        # self.noise_spectrum = tf.placeholder(dtype=tf.float32, shape=(
        #     self.batch_size, self.frame_size, self.target_spectra_size))  # 这里没用到

        # 带噪语音embedding的encoder
        _mix_LSTM_input = self.input_spectrum

        # (batch_size, frame_size, spectra_size)->(batch_size, frame_size, LSTM_weight_size)
        for i in range(self.layer_num_to_each_target):
            with tf.variable_scope('LSTM' + "_m_" + str(i + 1)):
                LSTM_cell = tf.contrib.rnn.LSTMCell(self.LSTM_weight_size)
                _mix_LSTM_input, _ = tf.nn.dynamic_rnn(LSTM_cell, _mix_LSTM_input, dtype=tf.float32)
                # _mix_LSTM_input = tf.nn.elu(_mix_LSTM_input)

        mean, var = tf.nn.moments(_mix_LSTM_input, [0, 1, 2])
        _mix_LSTM_input = tf.nn.batch_normalization(_mix_LSTM_input, mean, var, None, None, 0.001)

        # (batch_size, frame_size, LSTM_weight_size)->(batch_size, frame_size, spectra_size) 避免参数量过大
        _mix_LSTM_output = tf.layers.dense(_mix_LSTM_input, self.target_spectra_size)

        # (batch_size, frame_size, spectra_size)->(batch_size, frame_size, spectra_size*embedding_size)
        _mix_spectrum_embedding = tf.layers.dense(_mix_LSTM_output, self.target_spectra_size * self.embedding_size)
        # _mix_spectrum_embedding=tf.nn.elu(_mix_spectrum_embedding

        # 产生embedding阵列
        # (batch_size, frame_size, spectra_size*embedding_size)->(batch_size, frame_size, spectra_size, embedding_size)
        mix_spectrum_embedding_array = tf.nn.relu(
            tf.reshape(_mix_spectrum_embedding, [-1, self.frame_size, self.target_spectra_size, self.embedding_size]))
        # 归一化（这里归一化后存在负值）
        mix_spectrum_embedding_array = mix_spectrum_embedding_array / (
                tf.reduce_max(mix_spectrum_embedding_array) + np.spacing(1))

        # 纯净语音embedding的encoder （这里encoder与ASAM里的方式有所不同，测试过原始ASAM和修改版ASAM，效果接近）
        _clean_LSTM_input = self.clean_spectrum

        # (batch_size, frame_size, spectra_size)->(batch_size, frame_size, LSTM_weight_size)
        for i in range(self.layer_num_to_each_target):
            with tf.variable_scope('LSTM' + "_n_" + str(i + 1)):
                LSTM_cell = tf.contrib.rnn.LSTMCell(self.embedding_size * 3)
                _clean_LSTM_input, _ = tf.nn.dynamic_rnn(LSTM_cell, _clean_LSTM_input, dtype=tf.float32)
                # _clean_LSTM_input=tf.nn.elu(_clean_LSTM_input)

        mean, var = tf.nn.moments(_clean_LSTM_input, [0, 1, 2])
        _clean_LSTM_input = tf.nn.batch_normalization(_clean_LSTM_input, mean, var, None, None, 0.001)

        # (batch_size, frame_size, embedding_size * 3)->(batch_size, frame_size, spectra_size) 避免参数量过大
        _clean_LSTM_output = tf.layers.dense(_clean_LSTM_input, self.target_spectra_size)

        # (batch_size, frame_size, spectra_size)->(batch_size, frame_size, spectra_size*embedding_size)
        _clean_spectrum_embedding = tf.layers.dense(_clean_LSTM_output, self.target_spectra_size * self.embedding_size)

        # 产生embedding阵列
        # (batch_size, frame_size, spectra_size*embedding_size)->(batch_size, frame_size, spectra_size, embedding_size)
        clean_spectrum_embedding_array = tf.reshape(_clean_spectrum_embedding,
                                                    [-1, self.frame_size, self.target_spectra_size,
                                                     self.embedding_size])

        # 沿batch，时间，频率3个维度求平均，降维 (batch_size, frame_size, spectra_size, embedding_size)->(embedding_size)
        clean_spectrum_embedding1 = tf.reduce_mean(clean_spectrum_embedding_array, [0, 1, 2])

        clean_spectrum_embedding1 = tf.nn.relu(clean_spectrum_embedding1)

        # 归一化（这里多次迭代后，负值会消失）
        judge = tf.equal(clean_spectrum_embedding1, 0.)
        clean_spectrum_embedding_eps = tf.where(judge, tf.add(clean_spectrum_embedding1, np.spacing(1)),
                                                clean_spectrum_embedding1)
        clean_spectrum_embedding_eps = tf.sqrt(tf.reduce_sum(clean_spectrum_embedding_eps ** 2, axis=0, keep_dims=True))
        clean_spectrum_embedding1 = tf.div(clean_spectrum_embedding1, clean_spectrum_embedding_eps)

        # 将记忆向量加入长期记忆模块
        self.life_long_mem = tf.scatter_nd_add(self.life_long_mem, tf.constant([[0]]),
                                               tf.expand_dims(clean_spectrum_embedding1, 0))
        # 归一化
        self.life_long_mem = self.life_long_mem[0] / tf.norm(self.life_long_mem[0])

        # (batch_size, frame_size, spectra_size, embedding_size) 维度不变
        mix_spectrum_embedding_array, mask1 = self.embedding_array_update(mix_spectrum_embedding_array,
                                                                          self.life_long_mem)
        # embedding_array解码器
        if self.use_FC:
            # DNN解码器
            # (batch_size, frame_size, spectra_size, embedding_size)->(batch_size, frame_size, embedding_size*spectra_size)
            # (batch_size, frame_size, embedding_size*spectra_size)->(batch_size, frame_size, 1024)
            enhanced_spectrum = tf.layers.dense(
                tf.reshape(mix_spectrum_embedding_array,
                           [-1, self.frame_size, self.target_spectra_size * self.embedding_size]),
                1024)
            # (batch_size, frame_size, 1024)->(batch_size, frame_size, spectra_size)
            enhanced_spectrum = tf.layers.dense(
                enhanced_spectrum, self.target_spectra_size)
        else:
            # CNN解码器
            # (batch_size, frame_size, spectra_size, embedding_size)->(batch_size, frame_size, spectra_size,10)
            mix_spectrum_embedding_array = tf.nn.conv2d(mix_spectrum_embedding_array,
                                                        filter=tf.get_variable("f1", [3, 3, self.embedding_size, 10]),
                                                        strides=[1, 1, 1, 1], padding="SAME")
            # (batch_size, frame_size, spectra_size, embedding_size)->(batch_size, frame_size, spectra_size, 1)
            enhanced_spectrum = tf.nn.conv2d(mix_spectrum_embedding_array,
                                             filter=tf.get_variable("f11", [3, 3, 10, 1]), strides=[1, 1, 1, 1],
                                             padding="SAME")
            # (batch_size, frame_size, spectra_size, 1)->(batch_size, frame_size, spectra_size)
            enhanced_spectrum = tf.squeeze(enhanced_spectrum)

        loss = tf.nn.l2_loss(enhanced_spectrum - self.clean_spectrum)

        optimizer = tf.train.AdamOptimizer(0.001)
        step = optimizer.minimize(loss)

        # test用于测试和调试，对结果没有影响
        test = mask1
        return enhanced_spectrum, loss, step, self.life_long_mem, test
