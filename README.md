# Image captioning generation using MiniGPT-4 and Vicuna pre-trained model

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PythonAnywhere](https://img.shields.io/badge/pythonanywhere-%232F9FD7.svg?style=for-the-badge&logo=pythonanywhere&logoColor=151515)
![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white)


## Description

This repository constitutes an implementation of an **image captioner** for large datasets, aiming to streamline the creation process of **supervised datasets** to aid in the data augmentation procedure for image captioning deep learning architectures.

The foundational framework utilized is the [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), supplemented by the pre-trained [Vicuna](https://huggingface.co/Vision-CAIR/vicuna/tree/main) model boasting 13 billion parameters.

### Pre-requisite

You must have a GPU-enabled machine with a memory capacity of at least 23 GB.

## Getting Started

### Installation

```
git clone https://github.com/neemiasbsilva/MiniGPT-4-image-caption-implementation.git
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigptv
conda install pandas
mv MiniGPT-4/* ../.
```

### Setup the shell script

In the shell file (`run.sh`) you have to specify:
* `data_path`: the path where your image dataset are.
* `beam_search`: hyperparameter that is a range 0 to 10;
* `temperature`: hyperparameter (between 0.1 to 1.0);
* `save_path`: local you have to save your caption data set.

### Setup pre-trained models

* Download the [Vicuna 13 B](https://huggingface.co/Vision-CAIR/vicuna/tree/main)

* Set the LLM  path `minigpt4/configs/models/minigpt4_vicuna0.yaml` in Line 15.

    ```
    llama_model: "vicuna"
    ```

* Download the [MiniGPT-4 Checkpoint Model](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)

* Set the LLM  path `eval_configs/minigpt4_eval.yaml` in Line 8.
    ```
    ckpt: pretrained_minigpt4.pth
    ```

## Usage

```
sh run.sh
```