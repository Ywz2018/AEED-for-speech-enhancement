运行环境：
python=3.6
cuda=9.0
cudnn=7.6.0
tensorflow-gpu=1.13.1
numpy=1.17.0
librosa=0.7.0
soundfile=0.10.2

文件说明：
/EED/structure/direct/model_AEED.py :模型文件
/EED/execute/train/train_AEED.py :训练模型
/EED/execute/test/predict_AEED.py :评估模型

/config.py ：配置文件，
噪声和模型类型 ：NOISE_MODEL_TYPE
训练集、验证集、测试集的语音和噪声路径 ：
TRAIN_CLEAN_INPUT_PATH
TRAIN_NOISE_INPUT_PATH
VALID_CLEAN_INPUT_PATH
VALID_NOISE_INPUT_PATH
TEST_CLEAN_INPUT_PATH
TEST_NOISE_INPUT_PATH

数据集：
噪声：
Train:
Noisex92 :Babble, Factory1, Destroyerops, F16, White
Test:
Noisex92 :Babble, Factory1, F16, Factory2, Leopard, Hfchannel

语音：TIMIT