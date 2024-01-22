export CUDA_VISIBLE_DEVICES=$1
export OMP_NUM_THREADS=1

cfg=vicreg_patch.yml
exp_suffix=$2
epochs_1=200
#epochs_2=300

for sd in $3; do
  cfg=vicreg_patch.yml
  cmd_tr=$(echo main_pretrain.py -cfg ${cfg} --exp-suffix _sd${sd}${exp_suffix} --random-crop --infomin \
  --epochs ${epochs_1} --seed ${sd}) #  --optimizer adamw --base-lr 0.0002)
  echo "${cmd_tr}"
  python ${cmd_tr}

#  cfg=vicreg_patch.yml
#  cmd_tr=$(echo main_pretrain.py -cfg ${cfg} --exp-suffix _sd${sd}${exp_suffix} --epochs ${epochs_2} --seed ${sd}) # --optimizer adamw --base-lr 0.0002)
#  echo "${cmd_tr}"
#  python ${cmd_tr}

done
