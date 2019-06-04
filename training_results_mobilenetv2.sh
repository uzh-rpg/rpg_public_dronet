#! /bin/sh
#python cnn.py --experiment_rootdir=mobilenetv2_rgb_base --max_v_samples_per_dataset=1500 --max_t_samples_per_dataset=12500 --log_rate=1 --img_mode=rgb --epochs=50

python cnn.py --experiment_rootdir=mobilenetv2_rgb_dropout --max_v_samples_per_dataset=1500 --max_t_samples_per_dataset=12500 --log_rate=1 --img_mode=rgb --epochs=50 --dropout
python cnn.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout --max_v_samples_per_dataset=1500 --max_t_samples_per_dataset=12500 --log_rate=1 --img_mode=rgb --epochs=50 --hidden_dropout
#python cnn.py --experiment_rootdir=mobilenetv2_rgb_avg_pooling --max_v_samples_per_dataset=1500 --max_t_samples_per_dataset=12500 --log_rate=1 --img_mode=rgb --epochs=50 --pooling=avg

python cnn.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_aug --max_v_samples_per_dataset=1500 --max_t_samples_per_dataset=12500 --log_rate=1 --img_mode=rgb --epochs=50 --hidden_dropout --aug
#python cnn.py --experiment_rootdir=mobilenetv2_rgb_avg_pooling_aug --max_v_samples_per_dataset=1500 --max_t_samples_per_dataset=12500 --log_rate=1 --img_mode=rgb --epochs=50 --pooling=avg --aug
