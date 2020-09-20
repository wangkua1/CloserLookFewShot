
for method_name in protonet-ffs ; do
for model in Conv4; do
for lr in 1e-3 ; do

train_ffs=0
train_attr_split=train

case "$method_name" in
attr-train)
    method=attr
    train_attr_split=train
;;  
attr-all)
    method=attr
    train_attr_split=all
;;  
baseline)
    method=baseline
;; 
protonet)
    method=protonet
    train_ffs=0
;; 
protonet-ffs)
    method=protonet
    train_ffs=1
;;
maml)
    method=maml
    train_ffs=0
;; 
maml-ffs)
    method=maml
    train_ffs=1
;; 
esac

cmd=" eval.py \
	--checkpoint_dir saves/checkpoints/CUB-ffs0/${method_name}-${model}-${lr} \
	--dataset CUB \
	--model ${model} \
	--method ${method}  \
	--train_attr_split ${train_attr_split} \
	--train_ffs ${train_ffs} \
	--lr ${lr} 
"

if [ $1 == 0 ] 
then
python $cmd
else
sbatch <<< \
"#!/bin/bash
#SBATCH --mem=20G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=200:00:00
#SBATCH --output=/h/wangkuan/slurm/%j-out.txt
#SBATCH --error=/h/wangkuan/slurm/%j-err.txt
#SBATCH --qos=normal

#necessary env
source activate pnl

echo $cmd
python $cmd
"
fi

done
done
done