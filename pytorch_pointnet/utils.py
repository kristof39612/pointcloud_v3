import numpy as np
import matplotlib.pyplot as plt


def transform_2d_img_to_point_cloud(img):
    """
    Transforms a 2D image into a point cloud.
    Args:
        img (PIL.Image or numpy.ndarray): The input 2D image.
    Returns:
        numpy.ndarray: A 2D array of shape (N, 2) where N is the number of points
                       with intensity greater than 127. Each row represents the
                       normalized (x, y) coordinates of a point in the point cloud.
    """
    img_array = np.asarray(img)
    indices = np.argwhere(img_array > 127)
    for i in range(2):
        indices[i] = (indices[i] - img_array.shape[i]/2)/ img_array.shape[i]
    return indices.astype(np.float32)


def plot_losses(train_loss, test_loss, save_to_file=None):
    """
    Plots the training and test loss over epochs.
    Args:
        train_loss (list or array-like): A list or array of training loss values for each epoch.
        test_loss (list or array-like): A list or array of test loss values for each epoch.
        save_to_file (str, optional): If provided, the plot will be saved to the specified file path. Defaults to None.
    """
    fig = plt.figure()
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, 'bo', label='Training loss')
    plt.plot(range(epochs), test_loss, 'b', label='Test loss')
    plt.title('Training and test loss')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)


def plot_accuracies(train_acc, test_acc, save_to_file=None):
    """
    Plots the training and test accuracies over epochs.
    Args:
        train_acc (list or array-like): List of training accuracies for each epoch.
        test_acc (list or array-like): List of test accuracies for each epoch.
        save_to_file (str, optional): If provided, the plot will be saved to this file path. Defaults to None.
    """
    fig = plt.figure()
    epochs = len(train_acc)
    plt.plot(range(epochs), train_acc, 'bo', label='Training accuracy')
    plt.plot(range(epochs), test_acc, 'b', label='Test accuracy')
    plt.title('Training and test accuracy')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)
