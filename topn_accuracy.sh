echo "dronet_rgb_higher_l2 synthetic test Top-N=1" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=dronet_rgb_higher_l2_more_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=1
echo "dronet_rgb_higher_l2 synthetic test Top-N=2" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=dronet_rgb_higher_l2_more_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=2
echo "dronet_rgb_higher_l2 synthetic test Top-N=3" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=dronet_rgb_higher_l2_more_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=3
echo "dronet_rgb_higher_l2 synthetic test Top-N=4" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=dronet_rgb_higher_l2_more_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=4
echo "dronet_rgb_higher_l2 synthetic test Top-N=5" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=dronet_rgb_higher_l2_more_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=5

echo "\n\ndronet_rgb_higher_l2 real test Top-N=1" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=dronet_rgb_higher_l2_more_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=1
echo "\n\ndronet_rgb_higher_l2 real test Top-N=2" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=dronet_rgb_higher_l2_more_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=2
echo "dronet_rgb_higher_l2 real test Top-N=3" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=dronet_rgb_higher_l2_more_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=3
echo "dronet_rgb_higher_l2 real test Top-N=4" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=dronet_rgb_higher_l2_more_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=4
echo "dronet_rgb_higher_l2 real test Top-N=5" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=dronet_rgb_higher_l2_more_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=5

echo "\n\nmobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data synthetic test Top-N=1" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=1
echo "\n\nmobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data synthetic test Top-N=2" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=2
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data synthetic test Top-N=3" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=3
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data synthetic test Top-N=4" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=4
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data synthetic test Top-N=5" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=5

echo "\n\nmobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data real test Top-N=1" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=1
echo "\n\nmobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data real test Top-N=2" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=2
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data real test Top-N=3" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=3
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data real test Top-N=4" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=4
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data real test Top-N=5" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_noaugment_all_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=5

echo "\n\nmobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data synthetic test Top-N=1" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=1
echo "\n\nmobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data synthetic test Top-N=2" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=2
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data synthetic test Top-N=3" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=3
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data synthetic test Top-N=4" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=4
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data synthetic test Top-N=5" >> topn-accuracy-results.txt
python evaluation.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data --test_dir=/theos_dataset/test_synthetic --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=5

echo "\n\nmobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data real test Top-N=1" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=1
echo "\n\nmobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data real test Top-N=2" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=2
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data real test Top-N=3" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=3
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data real test Top-N=4" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=4
echo "mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data real test Top-N=5" >> topn-accuracy-results.txt
python testing.py --experiment_rootdir=mobilenetv2_rgb_hidden_dropout_avg_pooling_aug_all_data --test_dir=/media/theomorales/Theo/test_dataset_annotated --weights_fname=model_weights_99.h5 --img_mode=rgb --topn=5

