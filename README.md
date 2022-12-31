参考：https://github.com/LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch

训练的命令
python main.py --batch_size 24 --optimize_after_batches 2 --save_model | tee log.txt

测试的命令
python test.py

测试数据处理的命令
python dataloader.py