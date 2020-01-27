#!/bin/bash

dataset="dataset"
data_dir="$dataset"
results_dir="$dataset"
model_dir="$dataset/model"

trn_ft_file="${data_dir}/mydata.train.X"
trn_lbl_file="${data_dir}/mydata.train.y"
tst_ft_file="${data_dir}/mydata.test.X"
tst_lbl_file="${data_dir}/mydata.test.y"
score_file="${results_dir}/score_mat.txt"

# create the model folder
mkdir -p $model_dir

# training
# Reads training features (in $trn_ft_file), training labels (in $trn_lbl_file), and writes FastXML model to $model_dir
./parabel_train "/home/akash/Downloads/assn2/Parabel/dataset/mydata.train.X" "/home/akash/Downloads/assn2/Parabel/dataset/mydata.train.y" "/home/akash/Downloads/assn2/Parabel/dataset/model" -T 1 -s 0 -t 3 -b 1.0 -c 1.0 -m 100 -tcl 0.1 -tce 0 -e 0.0001 -n 20 -k 0 -q 0

# testing
# Reads test features (in $tst_ft_file), FastXML model (in $model_dir), and writes test label scores to $score_file
./parabel_predict $tst_ft_file $model_dir $score_file -t 3

# performance evaluation 
# matlab -nodesktop -nodisplay -r "cd('$PWD'); addpath(genpath('../Tools')); trn_X_Y = read_text_mat('$trn_lbl_file'); tst_X_Y = read_text_mat('$tst_lbl_file'); wts = inv_propensity(trn_X_Y,0.55,1.5); score_mat = read_text_mat('$score_file'); get_all_metrics(score_mat, tst_X_Y, wts); exit;"

