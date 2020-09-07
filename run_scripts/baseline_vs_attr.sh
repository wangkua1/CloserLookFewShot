
for method in attr baseline; do
for model in Conv4 Conv6 ResNet10 ResNet18 ResNet34; do

cmd="
python ./train.py --dataset CUB --model ${model} --method ${method}
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

# python ./train.py --dataset CUB --model Conv4 --method attr
# python ./train.py --dataset CUB --model Conv4 --method protonet