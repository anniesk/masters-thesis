#!/bin/bash
#SBATCH --job-name=toxicity
#SBATCH --account=project_2005092
#SBATCH --partition=gpu
#SBATCH --time=10:00:00 # 7h ?
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000 # 8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=grid_logs/%j.out
#SBATCH --error=logs/%j.err


# remove output and job marker on exit
# function on_exit {
#     rm -f out-$SLURM_JOBID.tsv
#     rm -f jobs/$SLURM_JOBID
# }
# trap on_exit EXIT

# # check arguments
# if [ "$#" -ne 4 ]; then
#     echo "Usage: $0 MODEL BATCH LR EPOCHS"
#     exit 1
# fi

# MODEL="$1"
# DATA="bretschneider2016en_lol.csv bretschneider2016en_wow.csv davidson2017en.csv founta2018en.csv novak2021sl.csv ousidhoum2019fr.csv ousidhoum2019en_with_stopwords.csv qian2019en_reddit.csv waseem2016en.csv zampieri2019en.csv"
# BATCH="$2"
# LR="$3"
# EPOCHS="$4"

# # symlink logs for this run as "latest"
# rm -f logs/latest.out logs/latest.err
# ln -s "$SLURM_JOBID.out" "grid_logs/latest.out"
# ln -s "$SLURM_JOBID.err" "grid_logs/latest.err"

# module purge
# module load pytorch

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# echo "START $SLURM_JOBID: $(date)"

# srun python train_models.py \
#     --files $DATA \
#     --model $MODEL \
#     --learning $LR \
#     --epochs $EPOCHS \
#     --batch $BATCH \
#     --jigsaw


# echo -n 'PARAMETERS'$'\t'
# echo -n 'model'$'\t'"$MODEL"$'\t'
# echo -n 'train_batch_size'$'\t'"$BATCH"$'\t'
# echo -n 'datasets'$'\t'"$DATA"$'\t'
# echo -n 'learning_rate'$'\t'"$LR"$'\t'
# echo -n 'num_train_epochs'$'\t'"$EPOCHS"$'\t'

# seff $SLURM_JOBID

# echo "END $SLURM_JOBID: $(date)"




echo "START $(date)"

module purge
module load pytorch 

EPOCHS=5 # 10 ?
LR=1e-5  
BATCH=8
MODEL="TurkuNLP/bert-base-finnish-cased-v1" #'bert-large-cased' # "xlm-roberta-large" #TurkuNLP/bert-base-finnish-cased-v1
echo "epochs: $EPOCHS, learning rate: $LR, batch size: $BATCH, model: $MODEL "

# davidson2017en.csv founta2018en.csv novak2021sl.csv ousidhoum2019fr.csv ousidhoum2019en_with_stopwords.csv waseem2016en.csv zampieri2019en.csv
DATA="bretschneider2016en_lol.csv bretschneider2016en_wow.csv qian2019en_reddit.csv"
# NOTE OUSIDHOUM I COUNTED AS ONE DATASET :D

SAVE_NAME="no_twitter"

# # #original english
# # echo "original english"
# # srun python3 toxic_classifier.py --train ../data/train_en.jsonl --test ../data/test_en.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --loss --dev --clean_as_label --save "og-bert-large3" #--threshold $TR


# # translated deepl
srun python3 train_models.py --files $DATA --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --save_name $SAVE_NAME --jigsaw 


# # transfer
# # echo "transfer with xlmr"
# # srun python3 toxic_classifier.py --train ../data/train_en.jsonl --test ../data/test_fi_deepl.jsonl --model $MODEL --batch $BATCH --epochs $EPOCHS --learning $LR --clean_as_label --loss --dev #--threshold $TR


echo "END: $(date)"