################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    pred_label = np.argmax(predictions, axis=1)
    ground_truth = targets
    accuracy = np.sum(np.equal(pred_label, targets)).mean()
    #print("accuracy is:", accuracy)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    score = []

    for data in data_loader:
        test_images, test_labels = data
        test_images_flat = test_images.reshape(test_labels.shape[0], 32*32*3)

        y_hat= model.forward(test_images_flat)
        score.append(accuracy(y_hat, test_labels))

        #print(score)

        avg_accuracy = np.mean(score)

     #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################


    trainset = cifar10_loader['train']
    testset = cifar10_loader['test']
    validset = cifar10_loader['validation']
    running_loss = 0
    graph_loss = []
    graph_acc = []
    epoch_accuracy = []
    loss = 0
    batch_vector = []
    val_accuracies = []

    # TODO: Initialize model and loss module
    model = MLP(32*32*3, hidden_dims, 10)
    loss_module = CrossEntropyModule()
    # TODO: Training loop including validation
    for epoch in range(0, epochs):
        epoch_loss = []
        avg_accuracy2 = []
        for images, labels in trainset:
            flat_image= images.reshape((batch_size, 32*32*3))
            #print("finished")

            "forward pass"
            y_hat= model.forward(flat_image)
            loss = loss_module.forward(y_hat, labels)

            "backward pass"
            model.backward(loss_module.backward(y_hat, labels))
            train_loss = CrossEntropyModule.forward(flat_image, y_hat, labels)
            train_accuracy = accuracy(y_hat, labels)
            #print(train_loss)


            for module in model.modules:
                #print(module)
                if hasattr(module, 'params'):
                    module.params['weight'] -= lr* module.grads['weight']
                    module.params['bias'] -= lr*module.grads['bias']

            epoch_loss.append(train_loss)
            avg_accuracy2.append(train_accuracy)

        running_loss = np.mean(epoch_loss)
        epoch_accuracy = np.mean(avg_accuracy2)
        print("length of running loss", len(epoch_loss), "length of epoch", len(avg_accuracy2))
        print(f"Epoch loss in {epoch} is : {running_loss}")
        print(f"Epoch accuracy at {epoch} is: {epoch_accuracy}")
        graph_loss.append((running_loss))
        graph_acc.append((epoch_accuracy))
        score = evaluate_model(model, validset)

        val_accuracies.append(score)
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.plot(graph_loss, 'g', label = "training_loss")
    ax2.plot(graph_acc, 'b', label = "accuracy")
    ax3.plot(val_accuracies, 'r', label = "validation")
    ax2.set_title("Veli's handsomeness over time")
    ax1.set_title("Jenn's existential luck over time")
    ax3.set_title("Validation accuracy")
    #plt.plot(graph_loss[1], graph_acc[0], graph_loss[0])
    plt.show()



    #val_accuracies = evaluate_model(model, validset)
    # TODO: Test best model
    test_accuracy = 0
    # TODO: Add any information you might want to save for plotting
    logging_dict = 0
    logging_info = []
    #######################
    # END OF YOUR CODE    #
    #######################
    print(val_accuracies)
    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments\]
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    