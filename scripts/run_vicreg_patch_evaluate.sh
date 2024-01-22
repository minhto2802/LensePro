export CUDA_VISIBLE_DEVICES=$1
export OMP_NUM_THREADS=1

exp_suffix=$2
sd=15

pretrained=experiments/vicreg_patch/wp_avg_str32_in_core_sd${sd}_infominV4/resnet50.pth

ng=5
fr=0.0  # 0.15
epoch=100  # 100
sd=1
ood_thr=60 #

for es in 100; do  #
  for sd in $3 ; do
    cfg=vicreg_patch_evaluate.yml
    cmd_tr=$(echo main_evaluate.py -cfg ${cfg} \
             --exp-suffix _sd${sd}_secondStage${exp_suffix}_es${es}_ng${ng}_ep${epoch}_ood${ood_thr} \
             --seed ${sd} --epochs ${epoch} --forget-rate ${fr} --num-gradual ${ng} \
             --ood-thr ${ood_thr} -es ${es} \
             --min-inv $4 --pretrained ${pretrained}  #
             )
    echo "${cmd_tr}"  #  --pretrained ${pretrained}
    python ${cmd_tr}
  done
done
