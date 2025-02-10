#!/bin/bash

# ex: ./scripts/run-nnunetextractor.sh /mnt/nnUNetv2 1 3d_fullre all /mnt/nnUNetv2/raw/Dataset001_BONBIDHIE/imagesTs /mnt/nnUNetv2/predicts/Dataset001_BONBIDHIE config.toml

if [ "$#" != 7 ]; then
    echo "usage: $0 [nnUNet_rootdir] [nnUNet_train_dataset] [nnUNet_config (3d_fullres)] [nnUNet_fold (1,2,3,4,5,all)] [input_dir] [nnUNet_output_dir] [nn-extractor config]"
    exit 255
fi

nnUNet_rootdir=$1
nnUNet_train_dataset=$2
nnUNet_config=$3
nnUNet_fold=$4
input_dir=$5
nnUNet_output_dir=$6
config=$7

# export nnUNet env variables.
export nnUNet_raw="${nnUNet_rootdir}/raw"
export nnUNet_preprocessed="${nnUNet_rootdir}/preprocessed"
export nnUNet_results="${nnUNet_rootdir}/results"

# list files based on _0000.nii.gz and run each file.
for filename in `ls ${input_dir}/*_0000.nii.gz`
do
    nnunet-extractor -i ${filename}  -o ${nnUNet_output_dir} -d ${nnUNet_train_dataset} -c ${nnUNet_config} -f ${nnUNet_fold} --save-probabilities -N nnunetv2 -C ${config}
done
