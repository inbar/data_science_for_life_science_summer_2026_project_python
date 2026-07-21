#!/bin/bash
#SBATCH --job-name=run_scoring
#SBATCH --chdir=../
#SBATCH --output=/home/%u/.logs/sbatch/run_scoring_%j.log
#SBATCH --partition=big
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@zedat.fu-berlin.de
#SBATCH --array=1

USER=$(whoami)

# Commands
# Run:
#   $ sbatch <params> run_scoring.sh <method>
#
# See all jobs:
#  $  SLURM_TIME_FORMAT=relative sacct -u <user> --format=jobid,start,end,elapsed,NCPUS,AllocCPUS,AveCPU,CPUTime,state
#
# See job status:
#  $  scontrol show job <job_id>


# Pass params like:
# $ METHOD=<method>,LEVEL=<level> sbatch run_scoring.sh 
# Example:
# $ METHOD=spearman sbatch run_scoring.sh
# $ METHOD=mi_ksg,K_NEIGHBORS=5,TAG=KNN_5 sbatch --cpus-per-task=8 --time=06:00:00 run_scoring.sh
#
# The METHOD variable is required
# All other variables are optional and have their default values defined above.
#
# Available methods:
# spearman
# partial_corr
# mi_ksg
# ig_mlp

SUBSAMPLE_SIZE="${SUBSAMPLE_SIZE:=-1}"
LEVEL="${LEVEL:="celltype.l2"}"
TEST_SPLIT_SIZE="${TEST_SPLIT_SIZE:=40}"
SEED="${SEED:=42}"
K_NEIGHBORS="${K_NEIGHBORS:=3}"
TAG="${TAG:=''}"

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


source setup_environment.sh
cd scripts

srun ./7_scoring.py --subsample_size $SUBSAMPLE_SIZE \
               --level $LEVEL \
               --test_split_size $TEST_SPLIT_SIZE \
               --seed $SEED \
               --k_neighbors $K_NEIGHBORS \
               --method $METHOD \
               --tag $TAG