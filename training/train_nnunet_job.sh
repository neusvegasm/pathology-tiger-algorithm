#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --gpus-per-node=1

echo "Removing data"
rm -rf $TMPDIR/data
mkdir $TMPDIR/data
echo "Copying data"
cp -r $HOME/nn_data_copy/nnUNet_raw $TMPDIR/data
cp -r $HOME/nn_data_copy/nnUNet_results $TMPDIR/data
cp -r $HOME/nn_data_copy/nnUNet_preprocessed $TMPDIR/data

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

pip3 install --upgrade pip
pip3 install nnunetv2
pip3 install lxml
pip3 install rtree

export nnUNet_raw="$TMPDIR/data/nnUNet_raw"
export nnUNet_results="$TMPDIR/data/nnUNet_results"
export nnUNet_preprocessed="$TMPDIR/data/nnUNet_preprocessed"

echo "Preprocessing"

nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -pl nnUNetPlannerResEncM

echo "Done preprocessing"

echo "Training"
nnUNetv2_train Dataset001_TIGER 2d all --npz -p nnUNetResEncUNetMPlans -device cuda

cp -r $TMPDIR/data/nnUNet_results $HOME/train_results_all
cp -r $TMPDIR/data/nnUNet_preprocessed $HOME/train_results_all