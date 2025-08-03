#!/bin/bash
#SBATCH --job-name=toxicity
#SBATCH --account=project_2005092
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # from 10 to 1
#SBATCH --mem-per-cpu=8000 # 8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=grid_logs/%j.out
#SBATCH --error=logs/%j.err


echo "START $(date)"

module purge
module load pytorch 


TOKENIZER="TurkuNLP/bert-base-finnish-cased-v1"


#DATA="all_reannotated.tsv"

# # ousidhoum2019fr.csv ousidhoum2019en_with_stopwords.csv jigsaw
# DATASET=("davidson2017en.csv" "founta2018en.csv" "bretschneider2016en_lol.csv" "bretschneider2016en_wow.csv" "novak2021sl.csv" "qian2019en_reddit.csv" "waseem2016en.csv" "zampieri2019en.csv")

# #no_ousidhoum no_jigsaw
# MODELS=("no_davidson" "no_founta" "no_lol" "no_wow" "no_novak" "no_qian" "no_waseem" "no_zampieri")
# #MODELS="all no_davidson no_founta no_jigsaw no_lol no_novak no_ousidhoum no_qian no_waseem no_wow no_zampieri"
# #MODELS="all"




# for i in "${!MODELS[@]}"; do
#     MODEL=${MODELS[$i]} # i or $i ?
#     DATA=${DATASET[$i]}
#     echo "model: $MODEL"
#     echo "data: $DATA"
#     srun python3 predict_eval.py --data $DATA --model $MODEL --tokenizer $TOKENIZER
#     echo ""
# done

# for MODEL in $MODELS ; do 
# # two loops might not work at the same time like this, previously only looped models with the same data
#     echo "model: $MODEL"
#     echo "data: $DATA"
#     srun python3 predict_eval.py --data $DATA --model $MODEL --tokenizer $TOKENIZER
#     echo ""
# done



MODEL="models/no_twitter" #"models/no_twitter" #"models/twitter_data" #"no_ousidhoum"
DATA= #"all_reannotated.tsv" #"ousidhoum2019fr.csv ousidhoum2019en_with_stopwords.csv"
echo "model: $MODEL"
echo "data: $DATA"
srun python3 predict_eval.py --data $DATA --model $MODEL --tokenizer $TOKENIZER



echo "END: $(date)"