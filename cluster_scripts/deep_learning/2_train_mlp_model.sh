#!/bin/bash
#SBATCH --job-name=train_mlp_model
#SBATCH --chdir=../../
#SBATCH --output=/home/%u/.logs/sbatch/train_mlp_model_%A_%a.log
#SBATCH --partition=big
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@zedat.fu-berlin.de

USER=$(whoami)

source setup_environment.sh
cd scripts/deep_learning

###
# Run with params:
# $ PARAM1=value PARAM2=value sbatch 2_train_mlp_model.sh

LEVEL="${LEVEL:="celltype.l2"}"
TEST_SPLIT_SIZE="${TEST_SPLIT_SIZE:=40}"
SEED="${SEED:=42}"
TAG="${TAG:=''}"
N_EPOCHS="${N_EPOCHS:=15}"

echo "Starting training task"
echo "====================="
echo "  Params:"
echo "    LEVEL: ${LEVEL}"
echo "    TEST_SPLIT_SIZE: ${TEST_SPLIT_SIZE}"
echo "    SEED: ${SEED}"
echo "    N_EPOCHS: ${N_EPOCHS}"
echo "    TAG: ${TAG}"

srun ./2_train.py \
             --level $LEVEL \
             --test_split_size $TEST_SPLIT_SIZE \
             --seed $SEED \
             --n_epochs $N_EPOCHS \
             --tag $TAG
