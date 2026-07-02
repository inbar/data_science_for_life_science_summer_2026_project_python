#!/bin/bash
#SBATCH --job-name run
#SBATCH --chdir /home/inbad03/workspace/data_science_for_life_science_summer_2026_project_python/scripts
#SBATCH --output /home/inbad03/.logs/sbatch/verify_data.%j.out
#SBATCH --partition=big
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --mem=10
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=poupik@gmail.com

USER=$(whoami)

SUBSAMPLE_SIZE=-1
LEVEL="celltype.l2"
TEST_SPLIT_SIZE=40
SEED=42
K_NEIGHBORS=3

METHOD_SPEARMAN="spearman"
METHOD_PARTIAL_CORRELATION="partial_corr"
METHOD_MI_KSG="mi_ksg"
METHOD_IG_MLP="ig_mlp"

LOG_FOLDER = "/home/${USER}/.logs/srun"

source setup_environment.sh
conda activate /home/${USER}/miniconda3/envs/data_science_in_life_sciences_project_2026_group_1

srun --ntasks=1 --cpus-per-task=1 echo "task A" --param1 > output_A.log 2>&1 &
srun --ntasks=1 --cpus-per-task=1 echo "task B" --param1 > output_A.log 2>&1 &
srun --ntasks=1 --cpus-per-task=1 echo "task C" --param1 > output_A.log 2>&1 &
srun --ntasks=1 --cpus-per-task=1 echo "task D" --param1 > output_A.log 2>&1 &

wait

echo "All jobs completed!"

## Spearman
#srun --ntasks=1 --cpus-per-task=1 ./5_scoring.py --subsample_size SUBSAMPLE_SIZE \
#               --level LEVEL \
#               --test_split_size TEST_SPLIT_SIZE \
#               --seed SEED \
#               --k_neighbors K_NEIGHBORS \
#               --method METHOD_SPEARMAN \
#               > "${LOG_FOLDER}/spearman_%j_.log" 2>&1 &
#
## Partial correlation
#srun --ntasks=1 --cpus-per-task=1 ./5_scoring.py --subsample_size SUBSAMPLE_SIZE \
#               --level LEVEL \
#               --test_split_size TEST_SPLIT_SIZE \
#               --seed SEED \
#               --k_neighbors K_NEIGHBORS \
#               --method METHOD_PARTIAL_CORRELATION \
#               > "${LOG_FOLDER}/spearman_%j_.log" 2>&1 &
#
## Mutual Information
#srun --ntasks=1 --cpus-per-task=1 ./5_scoring.py --subsample_size SUBSAMPLE_SIZE \
#               --level LEVEL \
#               --test_split_size TEST_SPLIT_SIZE \
#               --seed SEED \
#               --k_neighbors K_NEIGHBORS \
#               --method METHOD_MI_KSG \
#               > "${LOG_FOLDER}/spearman_%j_.log" 2>&1 &
#
## Integrated Gradient
#srun --ntasks=1 --cpus-per-task=1 ./5_scoring.py --subsample_size SUBSAMPLE_SIZE \
#               --level LEVEL \
#               --test_split_size TEST_SPLIT_SIZE \
#               --seed SEED \
#               --k_neighbors K_NEIGHBORS \
#               --method METHOD_IG_MLP \
#               > "${LOG_FOLDER}/spearman_%j_.log" 2>&1 &(base)