#!/bin/bash
#SBATCH --job-name=run_scoring
#SBATCH --chdir=../
#SBATCH --output=/home/%u/.logs/sbatch/run_scoring_%j.log
#SBATCH --partition=big
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@zedat.fu-berlin.de

USER=$(whoami)

# Commands
# Run:
#   $ sbatch <params> run_scoring.sh <method>
#
# See all jobs:
#  $  SLURM_TIME_FORMAT=relative sacct -u <user>  --format=jobid,start,end,elapsed,NCPUS,AllocCPUS,AveCPU,CPUTime,state
#
# See job status:
#  $  scontrol show job <job_id>


# Pass params like:
# $ PARAM1=val sbatch run_scoring.sh
# Example:
# $ TEST_SPLIT_SIZE=40 sbatch run_scoring.sh spearman
# $ sbatch --cpus-per-task=8 --time=06:00:00 run_scoring.sh mi_ksg
#
# The METHOD variable is required
# All other variables are optional and have their default values defined below.
#
# Available methods:
# spearman
# partial_corr
# mi_ksg
# ig_mlp


# Cheatsheet:
#
# TEST_SPLIT_SIZE=40 sbatch 2_run_scoring.sh spearman
# TEST_SPLIT_SIZE=40 sbatch 2_run_scoring.sh partial_corr
# TEST_SPLIT_SIZE=40 sbatch --cpus-per-task=8 --mem-per-cpu=4G 2_run_scoring.sh mi_ksg
# TEST_SPLIT_SIZE=40 MODEL_TAG=tuned sbatch 2_run_scoring.sh ig_mlp


METHOD=$1

SUBSAMPLE_SIZE="${SUBSAMPLE_SIZE:=-1}"
LEVEL="${LEVEL:="celltype.l2"}"
TEST_SPLIT_SIZE="${TEST_SPLIT_SIZE:=40}"
SEED="${SEED:=42}"
K_NEIGHBORS="${K_NEIGHBORS:=3}"
TAG="${TAG:=''}"
MODEL_TAG="${MODEL_TAG:=''}"


echo "Starting scoring task"
echo "====================="
echo "  Params:"
echo "    METHOD: ${METHOD}"
echo "    SUBSAMPLE_SIZE: ${SUBSAMPLE_SIZE}"
echo "    LEVEL: ${LEVEL}"
echo "    TEST_SPLIT_SIZE: ${TEST_SPLIT_SIZE}"
echo "    SEED: ${SEED}"
echo "    K_NEIGHBORS: ${K_NEIGHBORS}"
echo "    TAG: ${TAG}"
echo "    MODEL_TAG: ${MODEL_TAG}"


source setup_environment.sh
cd scripts

srun ./_5_scoring.py --subsample_size $SUBSAMPLE_SIZE \
               --level $LEVEL \
               --test_split_size $TEST_SPLIT_SIZE \
               --seed $SEED \
               --k_neighbors $K_NEIGHBORS \
               --method $METHOD \
               --tag $TAG \
               --model_tag $MODEL_TAG