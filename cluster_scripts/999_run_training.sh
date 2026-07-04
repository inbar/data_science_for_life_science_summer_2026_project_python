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

# Commands
# Run:
#   $ sbatch run_training.sh run_scoring.sh
#
# All jobs:
#  $  SLURM_TIME_FORMAT=relative sacct -u <user>  --format=jobid,start,end,elapsed,NCPUS,AllocCPUS,AveCPU,CPUTime,state
#
# Job status:
#  $  scontrol show job <job_id>

SUBSAMPLE_SIZE=-1
LEVEL="celltype.l2"
TEST_SPLIT_SIZE=40
SEED=42
N_EPOCHS=15

# Pass param like:
# $ sbatch run_training.sh some_tag
TAG=$1

echo "Starting scoring task"
echo "====================="
echo "  Params:"
echo "    SUBSAMPLE_SIZE: ${SUBSAMPLE_SIZE}"
echo "    LEVEL: ${LEVEL}"
echo "    TEST_SPLIT_SIZE: ${TEST_SPLIT_SIZE}"
echo "    SEED: ${SEED}"
echo "    N_EPOCHS: ${N_EPOCHS}"
echo "    TAG: ${TAG}"


echo "Starting training task"

source setup_environment.sh
cd scripts/deep_learning

./1_train.py --subsample_size $SUBSAMPLE_SIZE \
               --level $LEVEL \
               --test_split_size $TEST_SPLIT_SIZE \
               --seed $SEED \
               --n_epochs $N_EPOCHS \
               --tag $TAG