import os
import time
import numpy as np
import datetime
from src.models_gdlc_keras.model import Model
from keras import layers
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import timeseries_dataset_from_array
from keras.regularizers import l1, l2


class MyGRU(Model):
    def __init__(
            self,
            sequence_length,
            input_dim,
            dataset_name,
            stride=1,
            use_generator=False,
            already_sequenced=False,
            cells=[80],
            num_classes=2,
            multi_class=False,
            network_features=[],
            loss="mse",
            optimizer="adam",
            metrics=["accuracy"],
            batch_size=128,
            epochs=25,
            early_stop_patience=3
    ):

        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.dataset_name = dataset_name

        self.stride = stride
        self.use_generator = use_generator
        self.already_sequenced = already_sequenced
        self.cells = cells
        self.num_classes = num_classes
        self.multi_class = multi_class
        self.network_features = network_features
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience

        self.sequential = True
        self.model = Sequential()

    def model_name(self):
        return "gru"
        # classification = "bc"  # binary classification
        # if self.multi_class:
        #     classification = "mc"  # multi-class classification

        # network_features_string = ""
        # if self.network_features:
        #     network_features_string = "nf"
        #     for f in self.network_features:
        #         network_features_string += "-" + f

        # layers_name = ""
        # for layer in self.cells:
        #     layers_name += "-{}".format(layer)

        # return "gru sl-{} {} {} layers{}".format(self.sequence_length, classification, network_features_string, layers_name)

    def build(self):
        model = Sequential()

        # make dynamic
        # for i, c in enumerate(self.cells):
        #     if i == 0:
        #         model.add(layers.GRU(c, input_shape=(
        #     self.sequence_length, self.input_dim), return_sequences=False, kernel_regularizer=l2(0.01)))
        #     else:
        #         model.add(layers.GRU(c, return_sequences=False))
        # LAMBD = 3e-2                         # lambda in L2 regularizaion
        LAMBD = 0.01                         # lambda in L2 regularizaion
        print(f"==>> LAMBD: {LAMBD}")
        print(f"==>> datetime.datetime: {datetime.datetime}")

        # model.add(layers.GRU(units=80,
        #                      input_shape=(self.sequence_length,
        #                                   self.input_dim),
        #                      kernel_regularizer=l1(LAMBD),
        #                      recurrent_regularizer=l1(LAMBD),
        #                      bias_regularizer=l1(LAMBD),
        #                      return_sequences=True,
        #                      #  return_sequences=False,
        #                      # return_state=False,
        #                      dropout=0.2,
        #                      #  stateful=False, unroll=False
        #                      ))
        # model.add(layers.GRU(units=20,
        #                      input_shape=(self.sequence_length,
        #                                   self.input_dim),
        #                      kernel_regularizer=l1(LAMBD),
        #                      recurrent_regularizer=l1(LAMBD),
        #                      bias_regularizer=l1(LAMBD),
        #                      return_sequences=False,

        #                      # return_state=False,
        #                      dropout=0.2,
        #                      #  stateful=False, unroll=False
        #                      ))
        # model.add(layers.BatchNormalization())
        # model.add(layers.GRU(80, return_sequences=False))

        # model.add(layers.GRU(80, input_shape=(
        #     self.sequence_length, self.input_dim)))
        # model.add(layers.Normalization())

        model.add(layers.GRU(80, input_shape=(
            self.sequence_length, self.input_dim)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(100, activation='relu'))

        if self.multi_class:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            model.compile(
                optimizer=self.optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        else:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer=self.optimizer,
                          loss='binary_crossentropy', metrics=['accuracy'])

        print(self.input_dim)
        model.summary()
        self.model = model

    def create_generator(self, data, labels, batch_size):
        return timeseries_dataset_from_array(
            data=data,
            targets=labels[self.sequence_length - 1:],
            sequence_length=self.sequence_length,
            sequence_stride=self.stride,
            shuffle=False,
            batch_size=batch_size)

    def create_sequences(self, data, labels):
        data_reshaped = []
        labels_reshaped = []
        for i in range(self.sequence_length, len(data) + 1):
            data_reshaped.append(
                data[i - self.sequence_length:i, 0:data.shape[1]])
            # training_labels_reshaped.append(labels[i - seq_len:i])
        labels_reshaped = labels[self.sequence_length - 1:]

        data_reshaped = np.array(data_reshaped)
        labels_reshaped = np.array(labels_reshaped)

        return data_reshaped, labels_reshaped

    def train(self, training_data, training_labels, X_validation, y_validation):
        if self.model == None:
            self.build()

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stop_patience,
            restore_best_weights=True
        )
        checkpoint = ModelCheckpoint(
            filepath="temp/best_model_gru.keras",      # File path to save the model
            # Metric to monitor (e.g., validation loss)
            monitor='val_loss',
            verbose=1,                     # Verbosity mode, 1 = display messages
            save_best_only=True,           # Only save when the metric improves
            mode='min'                     # For "val_loss", lower is better
        )
        callbacks_list = [early_stopping, checkpoint]
        if self.already_sequenced:
            # print(f"==>> already_sequenced")
            print(f"==>> training_data.shape: {training_data.shape}")
            # print(f"==>> training_data.head: {training_data[:5]}")
            print(f"==>> training_labels.shape: {training_labels.shape}")
            # print(f"==>> training_labels.head: {training_labels[:5]}")
            history = self.model.fit(training_data, training_labels, epochs=self.epochs, validation_data=(X_validation, y_validation),
                                     batch_size=self.batch_size, shuffle=False, callbacks=callbacks_list)
        elif self.use_generator:
            training_generator = self.create_generator(
                training_data, training_labels, self.batch_size)
            validation_generator = self.create_generator(
                X_validation, y_validation, self.batch_size)
            history = self.model.fit(training_generator, epochs=self.epochs, validation_data=validation_generator,
                                     shuffle=False, callbacks=callbacks_list)
        else:
            if self.sequence_length == 1:
                x_train = np.reshape(
                    training_data, (training_data.shape[0], 1, training_data.shape[1]))
                y_train = training_labels
            else:
                x_train, y_train = self.create_sequences(
                    training_data, training_labels)

            history = self.model.fit(x_train, y_train, epochs=self.epochs, validation_data=(X_validation, y_validation),
                                     batch_size=self.batch_size, shuffle=False, callbacks=callbacks_list)
        return history

    def predict(self, testing_data):
        self.model = load_model("temp/best_model_gru.keras")
        start = time.time()
        testing_labels = np.zeros(testing_data.shape[0])
        if self.use_generator:
            testing_generator = self.create_generator(
                testing_data, testing_labels, self.batch_size)
            start = time.time()
            y_predictions = self.model.predict(testing_generator)
            # y_test = np.array(testing_labels[self.sequence_length - 1:])
        else:
            if self.sequence_length == 1:
                x_test = np.reshape(
                    testing_data, (testing_data.shape[0], 1, testing_data.shape[1]))
                y_test = testing_data
            else:
                x_test, y_test = self.create_sequences(
                    testing_data, testing_labels)
            start = time.time()
            y_predictions = self.model.predict(
                x_test, batch_size=self.batch_size)
        end = time.time()

        if self.num_classes == 2:

            y_predictions = np.transpose(y_predictions)[0]
            y_predictions = list(
                map(lambda x: 0 if x < 0.5 else 1, y_predictions))
            # y_predictions = (y_predictions >= 0.5).astype(int)
            # print(y_predictions)
        else:
            y_predictions = np.argmax(y_predictions, axis=1)

        # return a tupil of the predictions and the time it took to predict
        return (y_predictions, end-start)

    def evaluate(self, predictions, labels, time, labels_names, labels_dict, verbose=0):
        return super().evaluate(
            predictions,
            np.array(labels[self.sequence_length - 1:]),
            time,
            labels_names,
            labels_dict,
            verbose
        )
