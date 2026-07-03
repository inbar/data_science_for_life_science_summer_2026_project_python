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

SUBSAMPLE_SIZE=-1
LEVEL="celltype.l2"
TEST_SPLIT_SIZE=40
SEED=42
K_NEIGHBORS=3

# Pass param like:
# $ sbatch run_scoring.sh spearman
# $ sbatch run_scoring.sh partial_corr
# $ sbatch --cpus-per-task=8 --time=06:00:00 run_scoring.sh mi_ksg
# $ sbatch run_scoring.sh ig_mlp
METHOD=$1

echo "Starting scoring task: $METHOD"


source setup_environment.sh
cd scripts

srun ./5_scoring.py --subsample_size $SUBSAMPLE_SIZE \
               --level $LEVEL \
               --test_split_size $TEST_SPLIT_SIZE \
               --seed $SEED \
               --k_neighbors $K_NEIGHBORS \
               --method $METHOD