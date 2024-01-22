export CUDA_VISIBLE_DEVICES=$1
export OMP_NUM_THREADS=1

exp_suffix=$2
sd=15

#pretrained=experiments/vicreg_patch/wp_avg_str32_in_core_sd${sd}_infominV4_b_inProstate/resnet50.pth
pretrained=experiments/vicreg_patch/wp_avg_str32_in_core_sd${sd}_infominV4/resnet50.pth
#pretrained=experiments/vicreg_patch/wp_avg_str32_in_core_sd15_rr21/resnet50.pth
#pretrained=experiments/vicreg_patch/wp_avg_str32_in_core_sd15_rr22/resnet50.pth
#cfg=vicreg_patch_evaluate_semisup.yml

ng=5
fr=0.0  # 0.15
epoch=40  # 100
sd=1
ood_thr=60 #

for es in 100; do  #
#  for sd in 11; do
#  for sd in {0..9} ; do
  for sd in $3 ; do
  #  cfg=vicreg_p
  #  atch_evaluate_semisup.yml
  #  cmd_tr=$(echo vicreg_rf_patches_evaluate_coteaching_semisup.py -cfg ${cfg} \
  #           --exp-suffix _sd${sd}_secondStage${exp_suffix}_fr${fr}_ng${ng}_coteaching \
  #           --seed ${sd} --epochs 100 --pretrained ${pretrained} --forget-rate ${fr} --num-gradual ${ng} \
  #           --T .5 --threshold .0 --lambda-u 0 --mu 3
  #           )

#    cfg=vicreg_patch_evaluate.yml
#    cmd_tr=$(echo vicreg_rf_patches_evaluate_coteaching_gmm.py -cfg ${cfg} \
#             --exp-suffix _sd${sd}_secondStage${exp_suffix}_es${es}_ng${ng}_ep${epoch}_ood${ood_thr} \
#             --seed ${sd} --epochs ${epoch} --forget-rate ${fr} --num-gradual ${ng} \
#             --ood-thr ${ood_thr} -es ${es} \
#             --min-inv $4 #  --pretrained ${pretrained}  #
#             )
##    cmd_tr=$(echo vicreg_rf_patches_evaluate_coteaching.py -cfg ${cfg} \
##             --exp-suffix _sd${sd}_secondStage${exp_suffix}_es${es}_ng${ng}_ep${epoch}_ood${ood_thr} \
##             --seed ${sd} --epochs ${epoch} --pretrained ${pretrained} --forget-rate ${fr} --num-gradual ${ng}
##             )
#    echo "${cmd_tr}"  #  --pretrained ${pretrained}
##    python ${cmd_tr}
#    /h/minht/anaconda3/envs/torch_ipcai2024/bin/python ${cmd_tr}

    cfg=vicreg_patch_evaluate_dmix.yml
    cmd_tr=$(echo vicreg_rf_patches_evaluate_dmix.py -cfg ${cfg} \
             --exp-suffix _sd${sd}_secondStage${exp_suffix}_ep${epoch}_dmix \
             --seed ${sd} --epochs ${epoch}
#             --pretrained ${pretrained}
             )
    echo "${cmd_tr}"  #  --pretrained ${pretrained}
    /h/minht/anaconda3/envs/torch_ipcai2024/bin/python ${cmd_tr}

  done
done

#cmd_tr=$(echo vicreg_rf_patches_evaluate.py -cfg ${cfg} --exp-suffix _sd${sd}_secondStage${exp_suffix} --seed ${sd} --pretrained ${pretrained} --epochs 300)
#echo "${cmd_tr}"
#python ${cmd_tr}

#cmd_tr=$(echo vicreg_rf_patches_evaluate_coteaching.py -cfg ${cfg} --exp-suffix _sd${sd}_coteaching_from_scratch${exp_suffix} --seed ${sd} --epochs 300)
#echo "${cmd_tr}"
#python ${cmd_tr}

#cmd_tr=$(echo vicreg_rf_patches_evaluate.py -cfg ${cfg} --exp-suffix _sd${sd}_from_scratch${exp_suffix} --seed ${sd} --epochs 300)
#echo "${cmd_tr}"
#python ${cmd_tr}
