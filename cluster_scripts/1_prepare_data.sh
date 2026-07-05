#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --chdir=../
#SBATCH --output=/home/%u/.logs/sbatch/prepare_data_%j.log
#SBATCH --partition=big
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@zedat.fu-berlin.de

USER=$(whoami)

source setup_environment.sh

LEVEL="${LEVEL:="celltype.l2"}"
FORCE_RECREATE=${FORCE_RECREATE:=false}
TEST_SPLIT_SIZE="${TEST_SPLIT_SIZE:=40}"
SEED="${SEED:=42}"

echo "Starting prepare data task"
echo "=========================="
echo "  Params:"
echo "    LEVEL: ${LEVEL}"
echo "    FORCE_RECREATE: ${FORCE_RECREATE}"
echo "    TEST_SPLIT_SIZE: ${TEST_SPLIT_SIZE}"
echo "    SEED: ${SEED}"


##
# 1. Download the raw data
##
DATA_HOME="${PROJECT_HOME_ROOT}/.data_science_project"
DATA_FILE="GSE164378_RAW.tar"
METADATA_FILE="GSE164378_sc.meta.data_3P.csv.gz"

# Download if file doesn't exist yet
if [ ! -f "${DATA_HOME}/${DATA_FILE}" ]; then
  curl -o "${DATA_HOME}/${DATA_FILE}" https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/GSE164378_RAW.tar
fi

# Download if file doesn't exist yet
if [ ! -f "${DATA_HOME}/${METADATA_FILE}" ]; then
  curl -o "${DATA_HOME}/${METADATA_FILE}" https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164378/suppl/GSE164378_sc.meta.data_3P.csv.gz
fi


##
# 2. Create and persist full MuData dataset
##

cd scripts

echo "Load datast..."
srun --mem 24G ./1_load_full_dataset.py \
           --level $LEVEL \
           --force_recreate $FORCE_RECREATE

##
# 3. Split
##

echo "Split datast..."

srun ./3_split_dataset.py \
               --level $LEVEL \
               --test_split_size $TEST_SPLIT_SIZE \
               --seed $SEED

#####
##### 3. Feature selection
######
#
##echo "Feature selection..."
#
#srun ./4_feature_selection.py \
#               --level $LEVEL \
#               --test_split_size $TEST_SPLIT_SIZE \
#               --seed $SEED
#
#
echo "Done!"