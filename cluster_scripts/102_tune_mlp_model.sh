#!/bin/bash
#SBATCH --job-name=tune_mlp_model
#SBATCH --chdir=../
#SBATCH --partition=big
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/.logs/sbatch/tune_mlp_model_%A_%a.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@zedat.fu-berlin.de

USER=$(whoami)

source setup_environment.sh
cd scripts/deep_learning

###
# Run:
# $ sbatch 102_tune_mlp_model.sh

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

srun ./2_tune.py \
             --n_trials $N_TRIALS \
             --level $LEVEL \
             --test_split_size $TEST_SPLIT_SIZE \
             --seed $SEED
