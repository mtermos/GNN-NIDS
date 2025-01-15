import os
import time
import numpy as np
from src.models_gdlc_keras.model import Model
from keras import layers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import timeseries_dataset_from_array
from keras.utils import plot_model
# from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt


class MyCNN(Model):
    def __init__(
        self,
            input_dim,
            dataset_name,
            cnn=[64, 64],
            num_classes=2,
            multi_class=False,
            network_features=[],
            loss="mse",
            optimizer="adam",
            metrics=["accuracy"],
            batch_size=128,
            epochs=25,
            early_stop_patience=3,
    ):

        self.input_dim = input_dim
        self.dataset_name = dataset_name

        self.cnn = cnn
        self.multi_class = multi_class
        self.network_features = network_features
        self.num_classes = num_classes
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience

        self.model = Sequential()
        super(self.__class__, self).__init__()

    def model_name(self):
        return "cnn"
        # classification = "bc"  # binary classification
        # if self.multi_class:
        #     classification = "mc"  # multi-class classification

        # network_features_string = ""
        # if self.network_features:
        #     network_features_string = "nf"
        #     for f in self.network_features:
        #         network_features_string += "-" + f
        # cnn_layers = "cnn"
        # for layer in self.cnn:
        #     cnn_layers += "-{}".format(layer)

        # return "cnn {} {} {}".format(classification, network_features_string, cnn_layers)

    def build(self):
        model = Sequential()
        model.add(layers.Conv1D(64, kernel_size=3,  # int(self.input_dim / 100),
                  activation="relu", input_shape=(self.input_dim, 1)))

        model.add(layers.MaxPooling1D(2))
        # model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        # model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu'))
        if self.multi_class:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            model.compile(
                optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        else:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam',
                          loss='binary_crossentropy', metrics=['accuracy'])

        print(self.input_dim)
        print(model.summary())
        # plot_model(model, to_file='cnn_model.png',
        #            show_shapes=True, show_layer_names=True)

        # plot_model(model, to_file='C:\\Users\\Administrateur\\Desktop\\folder\\Python\\cnn_model.png',
        #            show_shapes=True, show_layer_names=True)
        # Display the image using matplotlib
        # img = plt.imread('cnn_model.png')
        # plt.imshow(img)
        # plt.show()
        self.model = model

    def train(self, training_data, training_labels, x_val, y_val):
        if self.model == None:
            self.build()
        if os.path.exists("./models/weights/" + self.dataset_name + "/" + self.model_name() + "/best.hdf5"):
            self.model.load_weights(
                "./models/weights/" + self.dataset_name + "/" + self.model_name() + "/best.hdf5")
        else:
            filepath = "./models/weights/" + self.dataset_name + "/" + \
                self.model_name() + \
                "/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
            checkpoint = ModelCheckpoint(
                filepath, verbose=1, save_best_only=False, mode='max')
            earlyStopping = EarlyStopping(
                monitor="loss", patience=self.early_stop_patience)
            callbacks_list = [checkpoint, earlyStopping]

            # X_train = np.reshape(training, (training.shape[0],training.shape[1],1))

            self.model.fit(training_data, training_labels, epochs=self.epochs,
                           validation_data=(x_val, y_val),
                           batch_size=self.batch_size, callbacks=callbacks_list)

    def predict(self, testing_data):
        start = time.time()
        # X_test = np.reshape(testing_data, (testing_data.shape[0],testing_data.shape[1],1))

        y_predictions = self.model.predict(
            testing_data, batch_size=self.batch_size)
        end = time.time()

        if self.multi_class:
            y_predictions = np.argmax(y_predictions, axis=1)
        else:
            y_predictions = np.transpose(y_predictions)[0]
            y_predictions = list(
                map(lambda x: 0 if x < 0.5 else 1, y_predictions))

        return (y_predictions, end-start)
