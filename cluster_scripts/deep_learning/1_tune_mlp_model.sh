#!/bin/bash
#SBATCH --job-name=tune_mlp_model
#SBATCH --chdir=../../
#SBATCH --output=/home/%u/.logs/sbatch/tune_mlp_model_%j.log
#SBATCH --partition=big
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@zedat.fu-berlin.de

USER=$(whoami)

source setup_environment.sh
cd scripts/deep_learning

LEVEL="${LEVEL:="celltype.l2"}"
TEST_SPLIT_SIZE="${TEST_SPLIT_SIZE:=40}"
SEED="${SEED:=42}"
N_TRIALS="${N_TRIALS:=50}"

echo "Starting scoring task"
echo "====================="
echo "  Params:"
echo "    LEVEL: ${LEVEL}"
echo "    TEST_SPLIT_SIZE: ${TEST_SPLIT_SIZE}"
echo "    SEED: ${SEED}"
echo "    N_TRIALS: ${N_TRIALS}"

srun ./1_tune.py \
             --n_trials $N_TRIALS \
             --level $LEVEL \
             --test_split_size $TEST_SPLIT_SIZE \
             --seed $SEED

echo done