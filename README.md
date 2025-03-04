# nn-extractor

## Getting Started

If you have a trained [nnUNetv2 model](https://github.com/MIC-DKFZ/nnUNet) and the testing images,
then you can extract the underlying information with:

```bash
# git clone and cd to the code directory.
git clone git@github.com:chhsiao1981/nn-extractor.git
cd nn-extractor

# install this package.
pip install -e .

# run scripts
./scripts/run-nnunetextractor.sh [nnUNet_rootdir] [nnUNet_train_dataset] [nnUNet_config (2d, 3d_fullres)] [nnUNet_fold (1,2,3,4,5,all)] [input_dir] [nnUNet_output_dir] [nn-extractor config]

# presented as web-api.
./scripts/dev_server.sh
```

example to run `./scripts/run-nnunetextractor.sh`:

```bash
./scripts/run-nnunetextractor.sh /mnt/nnUNetv2 1 3d_fullres all /mnt/nnUNetv2/raw/Dataset001_BONBIDHIE/imagesTs /mnt/nnUNetv2/predicts/Dataset001_BONBIDHIE config.toml
```


## Goal

Given a trained deep neural network model and inputs. `nn-extractor` extracts all the relevant information starting from inputs to output results. Currently we focus on 3D medical imaging segmentation tasks.

## Structure of the Extraction.

Given a deep neural network model and an input, the extracted information can be divided as the following categories:

* Input: the inputs.
* Preprocess: preprocessing information.
* Forward: snaphots of forward prediction process.
* Backward: snapshots of backward propagation process.
* Postprocess: postprocessing information.
* Output: the outputs.
* (Sub-)extractor: The recursive sub-extractors for sub-tasks.
* Taskflow: The sequence of the task-flow.
