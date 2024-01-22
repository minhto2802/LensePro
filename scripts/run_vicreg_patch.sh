export CUDA_VISIBLE_DEVICES=$1
export OMP_NUM_THREADS=1

cfg=vicreg_patch.yml
exp_suffix=$2
epochs_1=200
epochs_2=300
#epochs_val_1=300
#epochs_val_2=300

for sd in $3; do
  cfg=vicreg_patch.yml
  cmd_tr=$(echo vicreg_rf_patches.py -cfg ${cfg} --exp-suffix _sd${sd}${exp_suffix} --random-crop --infomin \
  --epochs ${epochs_1} --seed ${sd}) #  --optimizer adamw --base-lr 0.0002)
  echo "${cmd_tr}"
  python ${cmd_tr}

#  pretrained=experiments/vicreg_patch/wp_avg_str32_in_core_sd${sd}${exp_suffix}/resnet50.pth
#  cfg=vicreg_patch_evaluate.yml
#  cmd_tr=$(echo vicreg_rf_patches_evaluate_coteaching.py -cfg ${cfg} \
#  --exp-suffix _sd${sd}_1st_stage${exp_suffix}_coteaching \
#  --seed ${sd} --pretrained ${pretrained} --epochs ${epochs_val_2})
#  echo "${cmd_tr}"
#  python ${cmd_tr}

#  pretrained=experiments/vicreg_patch_search/wp_avg_str32_in_core_sd${sd}${exp_suffix}/resnet50.pth
#  cfg=vicreg_patch_evaluate.yml
#  fr=0.1
#  ng=20
#  epoch=200
#  cmd_tr=$(echo vicreg_rf_patches_evaluate_coteaching.py -cfg ${cfg} \
#           --exp-suffix _sd${sd}_firstStage${exp_suffix}_fr${fr}_ng${ng}_ep${epoch}_coteaching_20coef_12pro \
#           --seed ${sd} --epochs ${epoch} --pretrained ${pretrained} --forget-rate ${fr} --num-gradual ${ng}
#           )
#  echo "${cmd_tr}"  #  --pretrained ${pretrained}
#  python ${cmd_tr}

  cfg=vicreg_patch.yml
  cmd_tr=$(echo vicreg_rf_patches.py -cfg ${cfg} --exp-suffix _sd${sd}${exp_suffix} --epochs ${epochs_2} --seed ${sd}) # --optimizer adamw --base-lr 0.0002)
  echo "${cmd_tr}"
  python ${cmd_tr}

#  pretrained=experiments/vicreg_patch/wp_avg_str32_in_core_sd${sd}${exp_suffix}/resnet50.pth
#  cfg=vicreg_patch_evaluate.yml
#  cmd_tr=$(echo vicreg_rf_patches_evaluate_coteaching.py -cfg ${cfg} \
#  --exp-suffix _sd${sd}_2nd_stage${exp_suffix}_coteaching \
#  --seed ${sd} --pretrained ${pretrained} --epochs ${epochs_val_2})
#  echo "${cmd_tr}"
#  python ${cmd_tr}

#  pretrained=experiments/vicreg_patch_search/wp_avg_str32_in_core_sd${sd}${exp_suffix}/resnet50.pth
#  cfg=vicreg_patch_evaluate.yml
#  fr=0.1
#  ng=20
#  epoch=200
#  cmd_tr=$(echo vicreg_rf_patches_evaluate_coteaching.py -cfg ${cfg} \
#           --exp-suffix _sd${sd}_secondStage${exp_suffix}_fr${fr}_ng${ng}_ep${epoch}_coteaching_20coef_12pro \
#           --seed ${sd} --epochs ${epoch} --pretrained ${pretrained} --forget-rate ${fr} --num-gradual ${ng}
#           )
#  echo "${cmd_tr}"  #  --pretrained ${pretrained}
#  python ${cmd_tr}

done
