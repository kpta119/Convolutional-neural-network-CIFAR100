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

2. To evaluate the model, ensure the [load](http://_vscodecontentref_/1) variable is set to `True` in [script.py](http://_vscodecontentref_/2) and run:
    ```sh
    python script.py
    ```

## Project Structure

- [model_conv5.py](http://_vscodecontentref_/4): Defines the CNN model architecture.
- [cifar_dataloader.py](http://_vscodecontentref_/5): Handles data loading and transformations.
- [train_epoch.py](http://_vscodecontentref_/6): Contains the training loop for one epoch.
- [eval.py](http://_vscodecontentref_/7): Contains the evaluation logic for the model.
- [script.py](http://_vscodecontentref_/8): Main script for training and evaluating the model.
- [README.md](http://_vscodecontentref_/9): Project documentation.

## Training

To train the model, ensure the [load](http://_vscodecontentref_/10) variable is set to `False` in [script.py](http://_vscodecontentref_/11). The training process includes:

- Loading the CIFAR-100 dataset.
- Defining the CNN model architecture.
- Training the model for a specified number of epochs.
- Saving the trained model to a file.

## Evaluation

To evaluate the model, ensure the [load](http://_vscodecontentref_/12) variable is set to `True` in [script.py](http://_vscodecontentref_/13). The evaluation process includes:

- Loading the trained model from a file.
- Evaluating the model on the CIFAR-100 test dataset.
- Printing the evaluation metrics such as accuracy, precision, recall, and F1-score.
- Plotting the training and test losses over epochs.

## Results

The results of the model evaluation, including confusion matrices and performance metrics, will be displayed and saved as images.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
