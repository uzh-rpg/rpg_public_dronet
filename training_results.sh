#! /bin/sh
#
# training_results.sh
# Copyright (C) 2019 theomorales <theomorales@air-admin>
#
# Distributed under terms of the MIT license.

python cnn.py --experiment_rootdir=dronet_gray_base --max_v_samples_per_dataset=1500 --max_t_samples_per_dataset=12500 --log_rate=1 --img_mode=grayscale --epochs=50 --transfer_learning --model_transfer_fpath=dronet/model_struct.json --weights_fpath=dronet/best_weights.h5
python cnn.py --experiment_rootdir=dronet_rgb_base --max_v_samples_per_dataset=1500 --max_t_samples_per_dataset=12500 --log_rate=1 --img_mode=rgb --epochs=50 --transfer_learning --model_transfer_fpath=dronet/model_struct.json --weights_fpath=dronet/best_weights.h5

python cnn.py --experiment_rootdir=dronet_rgb_hidden_dropout --max_v_samples_per_dataset=1500 --max_t_samples_per_dataset=12500 --log_rate=1 --img_mode=rgb --epochs=50 --hidden_dropout --transfer_learning --model_transfer_fpath=dronet/model_struct.json --weights_fpath=dronet/best_weights.h5
python cnn.py --experiment_rootdir=dronet_rgb_higher_l2 --max_v_samples_per_dataset=1500 --max_t_samples_per_dataset=12500 --log_rate=1 --img_mode=rgb --epochs=50 --higher_l2 --transfer_learning --model_transfer_fpath=dronet/model_struct.json --weights_fpath=dronet/best_weights.h5
