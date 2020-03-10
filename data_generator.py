import librosa
import os
import numpy as np
import random
import soundfile as sf
import config
FRAME_RATE=16000
WINDOW=int(0.032 * FRAME_RATE)
# r = 1300000
class Generator(object):
    def __init__(self,speech_input_path,noise_input_path=None,noise_input_path2=None,noisy_input_path=None):
        self.noise_input_path=noise_input_path
        self.speech_input_path=speech_input_path
        self.noisy_input_path=noisy_input_path
        self.noise_input_path2=noise_input_path2


    def load_wav(self,audio_file_path):
        wav,sr=librosa.load(audio_file_path, None)
        if sr != 16000:
            wav=librosa.resample(wav,sr,16000)
        # print("sr",sr)
        return np.array(wav),sr

    def calc_spectrum(self,wav):
        return np.abs(librosa.stft(wav,512,256,WINDOW))
    
    def calc_real_spectrum(self,wav):
        return np.real(librosa.stft(wav,512,256,WINDOW))
    
    def calc_imag_spectrum(self,wav):
        return np.imag(librosa.stft(wav,512,256,WINDOW))
    
    def calc_phase(self,wav):
        return np.angle(librosa.stft(wav,512,256,WINDOW))

    def create_mixture(self,voice,noise,snr,short_noise=True):
        # 取固定的分段
        segment_lenth=59904
        mixture,voice,noise = self.add_and_align_wav(voice[:segment_lenth],noise,segment_lenth,snr=snr,short_noise=short_noise)
        return mixture,voice,noise

    def replicate(self,wav, n_increment):
        wav = list(wav)
        i = 0
        while i < n_increment:
            wav.append(wav[i])
            i += 1
        return np.array(wav)

    def create_2x_even_wav(self,wav):
        inverse_wav=wav[::-1]
        return np.concatenate([inverse_wav,np.array([0]),wav],0)

    def reform_wav_from_2x_even(self,wav):
        return wav[np.shape(wav)[0]//2+1:]

    def add_and_align_wav(self, voice, noise, lenth,snr,short_noise=True):
        # Train (0,1360000){(0,1359904)} Test(1360000,1700000)
        # 噪声或语音段：59904
        noise_type=config.NOISE_MODEL_TYPE
        global r # 这里r=1360000

        if config.MODE=="train":
            # print("noise type:",noise_type)
            # 训练集开始的噪声分段r=[0,1360000]
            r = 0 # 重置
            # 根据前面设定的噪声类型，选择训练噪声的分段方式，一般用"c"（完全随机选择噪声分段）
            if noise_type.startswith("s1300n"):
                r=random.randint(0,1300)*1000

            elif noise_type.startswith("s130n"):
                r=random.randint(0,130)*10000

            elif noise_type.startswith("s13000n"):
                r=random.randint(0,13000)*100

            elif noise_type.startswith("c"):
                r=random.randint(0,1300000)

            elif noise_type.startswith("step"):
                r=random.randint(0,21)*60000

            # print(r)
            if np.shape(voice)[0] > lenth:
                voice = voice[:lenth]
            else:
                voice=np.pad(voice,(0,lenth-np.shape(voice)[0]),"constant")
                # voice = self.replicate(voice, lenth - np.shape(voice)[0])

        elif config.MODE=="test":
            if not short_noise:
                # 如果选择长噪声，在测试集中1360000开始每次选择往后3000位置的噪声
                r+=3000
            else:
                # 短噪声不作分段后移，r不起作用
                pass
            print(r)
            if np.shape(voice)[0] > lenth:
                voice = voice[:lenth]
            else:
                voice=np.pad(voice,(0,lenth-np.shape(voice)[0]),"constant")
                # voice = self.replicate(voice, lenth - np.shape(voice)[0])

        if short_noise:
            if np.shape(noise)[0]<lenth:
                # 短噪声需加长，使其长度达到语音长度
                noise=np.concatenate([noise]*4,0)[:lenth]
            else:
                noise=noise[:lenth]

        else:
            if np.shape(noise)[0]>lenth:
                noise=noise[r:r+lenth]
            else:
                noise=np.pad(noise,(0,lenth-np.shape(noise)[0]),"constant")

        # 根据所需信噪比缩放噪声幅值
        n_2 = np.sum(noise ** 2)
        if n_2 > 0.:
            if snr != None:
                a = np.sqrt(np.sum(voice ** 2) / (n_2 * 10 ** (snr / 10)))
                noise = a * noise

        mixture = voice + noise
        return mixture,voice,noise

    # 直接截取每一个短噪声到合适的分段，噪聲語音對應相加
    def prepare_data_att_PL_tiny(self,snr=-5,forward_step=0,short_noise=True,noise_in_use=1984,start_index=0,random_noise=False):
        speech_list = os.listdir(self.speech_input_path)
        noise_list = os.listdir(self.noise_input_path)[start_index:start_index+noise_in_use]
        # 改變列表中噪聲順序
        for i in range(forward_step):
            noise=noise_list.pop(0)
            noise_list.append(noise)

        if len(speech_list)<128:
            a=128//len(speech_list)+1
            speech_list=speech_list*a
            speech_list=speech_list[:128]

        if len(noise_list)<len(speech_list):
            a=int(len(speech_list)/len(noise_list))+1
            noise_list=noise_list*a
            noise_list=noise_list[:len(speech_list)]
        else:
            a = int(len(noise_list) / len(speech_list)) + 1
            speech_list = speech_list * a
            speech_list = speech_list[:len(noise_list)]

        if random_noise:
            random.shuffle(noise_list)

        mix_spectrum_batch = []
        noise_spectrum_batch = []
        speech_spectrum_batch = []
        mix_phase_batch = []
        voice_batch = []
        mixture_batch = []
        voice_lenth_batch=[]
        k = 0

        # r:测试数据开始取的位置
        global r
        if short_noise:
            r=0
        else:
            r=1360000

        start_snr=snr
        for speech_audio,noise_audio in zip(speech_list,noise_list):

            speech_audio_path = os.path.join(self.speech_input_path, speech_audio)
            noise_audio_path = os.path.join(self.noise_input_path, noise_audio)
            # print(speech_audio)
            # print(snr)

            speech_wav, sr = self.load_wav(speech_audio_path)
            noise_wav, sr = self.load_wav(noise_audio_path)

            mixture, voice, noise = self.create_mixture(speech_wav, noise_wav, snr, short_noise=short_noise)

            if config.MODE=="train":
                # 改变合成信噪比
                snr+=1
                # 控制信噪比范围
                if snr > start_snr + 5:
                    snr = start_snr

            mix_spectrum = self.calc_spectrum(mixture)
            speech_spectrum = self.calc_spectrum(voice)
            noise_spectrum = self.calc_spectrum(noise)
            mix_phase = self.calc_phase(mixture)

            mix_spectrum_batch.append(mix_spectrum)
            speech_spectrum_batch.append(speech_spectrum)
            noise_spectrum_batch.append(noise_spectrum)
            mix_phase_batch.append(mix_phase)
            voice_batch.append(voice)
            mixture_batch.append(mixture)
            voice_lenth_batch.append(np.shape(speech_wav)[0])

            if len(mix_spectrum_batch) == 32:
                k += 1
                print(str(k))
                mix_spectrum_batch = np.transpose(np.array(mix_spectrum_batch), [0, 2, 1])
                speech_spectrum_batch = np.transpose(np.array(speech_spectrum_batch), [0, 2, 1])
                noise_spectrum_batch = np.transpose(np.array(noise_spectrum_batch), [0, 2, 1])
                mix_phase_batch = np.transpose(np.array(mix_phase_batch), [0, 2, 1])
                voice_batch = np.array(voice_batch)
                mixture_batch = np.array(mixture_batch)

                yield mix_spectrum_batch, noise_spectrum_batch, speech_spectrum_batch, mix_phase_batch, voice_batch, mixture_batch,voice_lenth_batch
                mix_spectrum_batch = []
                noise_spectrum_batch = []
                speech_spectrum_batch = []
                mix_phase_batch = []
                voice_batch = []
                mixture_batch = []
                voice_lenth_batch=[]

    # 一般AEED用的
    # 对长噪声随机分段
    def prepare_data_att_PL_simple(self,snr=-5,chose_noise=None):
        noise_list = os.listdir(self.noise_input_path)
        speech_list = os.listdir(self.speech_input_path)

        mix_spectrum_batch = []
        noise_spectrum_batch = []
        speech_spectrum_batch = []
        mix_phase_batch=[]
        voice_batch = []
        mixture_batch = []
        voice_lenth_batch=[]
        k=0
        # 语音数量不够，复制，增加
        if len(speech_list)<128:
            a=128//len(speech_list)+1
            speech_list=speech_list*a
            speech_list=speech_list[:128]
        # noise_audio = noise_list[0]

        start_snr = snr
        for noise_audio in noise_list:
            # 没遇到需要的噪声就跳过
            if chose_noise:
                if noise_audio.split(".")[0]!=chose_noise:
                    continue

            noise_audio_path = os.path.join(self.noise_input_path, noise_audio)
            noise_wav, sr = self.load_wav(noise_audio_path)

            # 对长噪声分段的标签 Noisex92有大约3分钟，0为训练开始的标签
            global r
            # 1360000为测试集开始的标签
            r = 1360000
            print(noise_audio)

            for speech_audio in speech_list:
                speech_audio_path = os.path.join(self.speech_input_path, speech_audio)
                speech_wav,sr = self.load_wav(speech_audio_path)

                mixture,voice,noise = self.create_mixture(speech_wav, noise_wav, snr=snr,short_noise=False)
                if config.MODE == "train":
                    # 改变合成信噪比
                    snr += 1
                    # 控制信噪比范围
                    if snr > start_snr + 8:
                        snr = start_snr

                mix_spectrum = self.calc_spectrum(mixture)
                speech_spectrum = self.calc_spectrum(voice)
                noise_spectrum = self.calc_spectrum(noise)
                mix_phase = self.calc_phase(mixture)

                mix_spectrum_batch.append(mix_spectrum)
                speech_spectrum_batch.append(speech_spectrum)
                noise_spectrum_batch.append(noise_spectrum)
                mix_phase_batch.append(mix_phase)
                voice_batch.append(voice)
                mixture_batch.append(mixture)
                voice_lenth_batch.append(np.shape(speech_wav)[0])

                if len(mix_spectrum_batch)==32:
                    k+=1
                    print(str(k))
                    mix_spectrum_batch=np.transpose(np.array(mix_spectrum_batch),[0,2,1])
                    speech_spectrum_batch=np.transpose(np.array(speech_spectrum_batch),[0,2,1])
                    noise_spectrum_batch=np.transpose(np.array(noise_spectrum_batch),[0,2,1])
                    mix_phase_batch=np.transpose(np.array(mix_phase_batch),[0,2,1])
                    voice_batch=np.array(voice_batch)
                    mixture_batch=np.array(mixture_batch)

                    yield mix_spectrum_batch, noise_spectrum_batch, speech_spectrum_batch,mix_phase_batch,voice_batch,mixture_batch,voice_lenth_batch
                    mix_spectrum_batch = []
                    noise_spectrum_batch = []
                    speech_spectrum_batch = []
                    mix_phase_batch=[]
                    voice_batch=[]
                    mixture_batch=[]
                    voice_lenth_batch = []

    # 对长噪声随机分段
    def prepare_data_real_spectrum_simple(self,snr=-5,chose_noise=None):
        noise_list = os.listdir(self.noise_input_path)
        speech_list = os.listdir(self.speech_input_path)

        mix_spectrum_batch = []
        noise_spectrum_batch = []
        speech_spectrum_batch = []
        # mix_phase_batch=[]
        voice_batch = []
        mixture_batch = []
        voice_lenth_batch=[]
        k=0
        if len(speech_list)<128:
            a=128//len(speech_list)+1
            speech_list=speech_list*a
            speech_list=speech_list[:128]
        # noise_audio = noise_list[0]

        start_snr = snr
        for noise_audio in noise_list:
            if chose_noise:
                if noise_audio.split(".")[0]!=chose_noise:
                    continue

            noise_audio_path = os.path.join(self.noise_input_path, noise_audio)
            noise_wav, sr = self.load_wav(noise_audio_path)
            global r
            r = 1360000
            print(noise_audio)

            for speech_audio in speech_list:
                speech_audio_path = os.path.join(self.speech_input_path, speech_audio)
                speech_wav,sr = self.load_wav(speech_audio_path)

                mixture,voice,noise = self.create_mixture(speech_wav, noise_wav, snr=snr,short_noise=False)
                mixture=self.create_2x_even_wav(mixture)
                voice=self.create_2x_even_wav(voice)
                noise=self.create_2x_even_wav(noise)

                if config.MODE == "train":
                    # 改变合成信噪比
                    snr += 1
                    # 控制信噪比范围
                    if snr > start_snr + 8:
                        snr = start_snr

                mix_spectrum = self.calc_spectrum(mixture)
                speech_spectrum = self.calc_spectrum(voice)
                noise_spectrum = self.calc_spectrum(noise)
                # mix_phase = self.calc_phase(mixture)

                mix_spectrum_batch.append(mix_spectrum)
                speech_spectrum_batch.append(speech_spectrum)
                noise_spectrum_batch.append(noise_spectrum)
                # mix_phase_batch.append(mix_phase)
                voice_batch.append(voice)
                mixture_batch.append(mixture)
                voice_lenth_batch.append(np.shape(speech_wav)[0])

                if len(mix_spectrum_batch)==10:
                    k+=1
                    print(str(k))
                    mix_spectrum_batch=np.transpose(np.array(mix_spectrum_batch),[0,2,1])
                    speech_spectrum_batch=np.transpose(np.array(speech_spectrum_batch),[0,2,1])
                    noise_spectrum_batch=np.transpose(np.array(noise_spectrum_batch),[0,2,1])
                    # mix_phase_batch=np.transpose(np.array(mix_phase_batch),[0,2,1])
                    voice_batch=np.array(voice_batch)
                    mixture_batch=np.array(mixture_batch)

                    yield mix_spectrum_batch, noise_spectrum_batch, speech_spectrum_batch,voice_batch,mixture_batch,voice_lenth_batch
                    mix_spectrum_batch = []
                    noise_spectrum_batch = []
                    speech_spectrum_batch = []
                    # mix_phase_batch=[]
                    voice_batch=[]
                    mixture_batch=[]
                    voice_lenth_batch = []
