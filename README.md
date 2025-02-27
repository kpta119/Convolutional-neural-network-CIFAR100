# Convolutional Neural Network for CIFAR-100

This project implements a Convolutional Neural Network (CNN) for classifying images from the CIFAR-100 dataset. The project includes training, evaluation, and visualization of the model's performance.

## Table of Contents

- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)


## Usage

1. To train the model, run:
    ```sh
    python script.py
    ```

2. To evaluate the model, ensure the `load` variable is set to `True` (if you want to evaluate written model in file) in [script.py](https://github.com/kpta119/Convolutional-neural-network-CIFAR100/blob/master/script.py) and `epochs` variable is set to `0` in order to only evaluate model and run:
    ```sh
    python script.py
    ```

## Project Structure

- [model_conv5.py](https://github.com/kpta119/Convolutional-neural-network-CIFAR100/blob/master/model_conv5.py): Defines the CNN model architecture.
- [cifar_dataloader.py](https://github.com/kpta119/Convolutional-neural-network-CIFAR100/blob/master/cifar_dataloader.py): Handles data loading and transformations.
- [train_epoch.py](https://github.com/kpta119/Convolutional-neural-network-CIFAR100/blob/master/train_epoch.py): Contains the training loop for one epoch.
- [eval.py](https://github.com/kpta119/Convolutional-neural-network-CIFAR100/blob/master/eval.py): Contains the evaluation logic for the model.
- [script.py](https://github.com/kpta119/Convolutional-neural-network-CIFAR100/blob/master/script.py): Main script for training and evaluating the model.
- [README.md](https://github.com/kpta119/Convolutional-neural-network-CIFAR100/blob/master/README.md): Project documentation.

## Training

To train the model, ensure the `save` variable is set to `Ture` in [script.py](https://github.com/kpta119/Convolutional-neural-network-CIFAR100/blob/master/script.py) to save the model to file after training. The training process includes:

- Loading the CIFAR-100 dataset.
- Defining the CNN model architecture.
- Training the model for a specified number of epochs.
- Saving the trained model to a file.

## Evaluation

To evaluate the model, ensure the `load` variable is set to `True` (if you want to evaluate written model in file) in [script.py](https://github.com/kpta119/Convolutional-neural-network-CIFAR100/blob/master/script.py). The evaluation process includes:

- Loading the trained model from a file.
- Evaluating the model on the CIFAR-100 test dataset.
- Printing the evaluation metrics such as accuracy, precision, recall, and F1-score.
- Plotting the training and test losses over epochs.

## Results

The results of the model evaluation, including confusion matrices and performance metrics, will be displayed and saved as images. Matrixes for the best trained model are shown in confusion_matrixes_1st_model folder.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
