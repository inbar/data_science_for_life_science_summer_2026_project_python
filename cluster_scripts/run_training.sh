#!/bin/bash
#SBATCH --job-name=run_training
#SBATCH --chdir=../
#SBATCH --output=/home/%u/.logs/sbatch/run_training_%j.log
#SBATCH --partition=big
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=15GB
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@zedat.fu-berlin.de
#SBATCH --array=1

USER=$(whoami)

SUBSAMPLE_SIZE=-1
LEVEL="celltype.l2"
TEST_SPLIT_SIZE=40
SEED=42
N_EPOCHS=15

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_FOLDER}/training_${TIMESTAMP}.log"

echo "Starting training task"

source setup_environment.sh
cd scripts/deep_learning

./1_train.py --subsample_size $SUBSAMPLE_SIZE \
               --level $LEVEL \
               --test_split_size $TEST_SPLIT_SIZE \
               --seed $SEED \
               --n_epochs $N_EPOCHS