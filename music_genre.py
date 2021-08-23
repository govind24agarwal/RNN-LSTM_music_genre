import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "/home/govind/Documents/ML/Velario Youtube/extracting_mfccs_music_genre/data.json"


def load_data(data_path):
    """Loads training dataset from json file

    Args:
        data_path (str): path to json file
    Returns:
        x (ndarray) : Inputs
        y (ndarray) : Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return x, y


def plot_history(history):
    """Plots  accuracy/loss for training/validation set as a function of epochs

    Args:
        history (object): Training history of model
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="upper right")
    axs[0].set_title("Accuracy eval")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].plot(history.history["val_loss"], label="test loss")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="lower right")
    axs[1].set_title("loss eval")


def prepare_datasets(test_size=0.25, validation_size=0.2):
    """Loads data and splits it into train, test and validation sets

    Args:
        test_size (float, optional): Value in [0,1] indicating ratio of dataset to be used as test set. Defaults to 0.25.
        validation_size (float, optional): Value in [0,1].indicating ratio of rest of the dataset to be used as validation set Defaults to 0.2.
    Returns:
        x_train (ndarray): Inputs used for training
        x_validation (ndarray): Inputs used for validation
        x_test (ndarray): Inputs used for testing
        y_train (ndarray): Targets used for training
        y_validation (ndarray): Targets used for validation
        y_test (ndarray): Targets used for testing
    """

    # load data
    x, y = load_data(DATA_PATH)

    # create train, test and validation split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train, y_train, test_size=validation_size)

    return x_train, x_test, x_validation, y_train, y_test, y_validation


def build_model(input_shape):
    """Generates RNN-LSTM model

    Args:
        input_shape (tuple)): Shape of input set
    Returns:
        model : RNN-LSTM model
    """


def predict(model, x, y):
    """Predict a single sample using the trained model

    Args:
        model : Trained model
        x ([type]): Input data
        y ([type]): Target
    """

    # add a dimension to input data for sample
    x = x[np.newaxis, ...]

    # perform prediction
    prediction = model.predict(x)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


if __name__ == "__main__":
    # get  train, validation, test splits
    x_train, x_test, x_validation, y_train, y_test, y_validation = prepare_datasets(
        0.25, 0.2)

    # create network
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(input_shape=input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    # train model
    history = model.fit(x_train, y_train, validation_data=(
        x_validation, y_validation), batch_size=32, epochs=30)

    # plot accuraccy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print("\nTest accuracy:", test_acc)
    print("\nTest loss:", test_loss)

    # predict sample
    predict(model, x_test[9], y_test[9])  # can input any data here
