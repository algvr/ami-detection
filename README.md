# Analysis of Electrocardiograms via Artificial Neural Networks For a Reliable Assessment of a Possible Myocardial Infarction

This is the official repository for our [DSTSES 2022 project](https://coss.ethz.ch/education/DSTSES.html), containing the code for the generation of the PTB-V dataset and the derivative segmentation map dataset, as well as the definition and training of the segmentation and classification networks.
Below, we describe the steps to reproduce our results and work with our framework.

## Setup

- Install the Python requirements in `requirements.txt`. We recommend using Python 3.9.
- Run `download_datasets.sh`. This will download the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.1/), which takes 3GB of disk space.
  Furthermore, a pretrained U-Net model will be downloaded, which can e.g. be used to generate the derivative segmentation map dataset from PTB-V without having to train a segmentation model from scratch.

## Generating the PTB-V Dataset

- Inspect `dataset_generation/generate_ptb_v_dataset.py` to change any configuration you may wish to change, then run this script.

## Training the Segmentation Model

- Generate the PTB-V dataset, as mentioned above. 
- Open `dataset_generation/data_augmentation.ipynb`, run the cell setting `save_dir`, then run the last cell to generate individual lead images from the PTB-V dataset.
- Run `segmentation/train_segmentation_model.sh` to train a U-Net segmentation model on the resulting individual lead image dataset using the same configuration we used. See the docstrings of the files within the `segmentation/` directory if you wish to train a custom segmentation model. Checkpoints will be created periodically in the `checkpoints` directory.

## Generating the Segmentation Image Dataset for Training the Classification Model

- Inspect `dataset_generation/generate_dataset_for_classification.py` to change any configuration you may wish to change, then run this script. This will generate a dataset of channel-separated ECG segmentation images using the specified segmentation model and the generated PTB-V dataset.
- Note that due to the probabilistic nature of our dataset generation process and the difficulty of finding perfect parameters for the data augmentation distributions used by this process, some generated PTB-V samples may possess extreme artifacts such as strong noise or shadows, which may prevent the segmentation network from correctly segmenting them. Such samples are unlikely to be submitted to the classification algorithm by real users. It is worth inspecting the generated channel-separated ECG segmentation images and removing any that appear distorted to ensure the soon-to-be-trained classification model receives an informative training set.
- Inspect and run `dataset_generation/classification_dataset_augmentation.py` to augment the resulting channel-separated segmentation image dataset.
- Inspect and run `dataset_generation/merge_classification_dataset_channels.py` to merge image files corresponding to the individual channels of the same (augmented) ECG sample to one image.

## Training the Classification Model
- Inspect `classification/train_classification_model.py` to change any configuration you may wish to change, then run this script.
