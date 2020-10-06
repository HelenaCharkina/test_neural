import time

import idx2numpy
import numpy as np
import tf as tf
from tensorflow import keras

from Neural import emnist_labels


def train_model(model):
    t_start = time.time()

    emnist_path = 'emnist/database/'


    X_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-images-idx3-ubyte')
    y_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-labels-idx1-ubyte')

    X_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-images-idx3-ubyte')
    y_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-labels-idx1-ubyte')

    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

    # Test:
    k = 9
    X_train = X_train[:X_train.shape[0] // k]
    y_train = y_train[:y_train.shape[0] // k]
    X_test = X_test[:X_test.shape[0] // k]
    y_test = y_test[:y_test.shape[0] // k]

    # Normalize
    X_train = X_train.astype(np.float32)
    X_train /= 255.0
    X_test = X_test.astype(np.float32)
    X_test /= 255.0

    y_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
    y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))

    model.compile(optimizer='adam',  # Optimizer
                  # Минимизируемая функция потерь
                  loss=keras.losses.CategoricalCrossentropy(),
                  # Список метрик для мониторинга
                  metrics=[keras.metrics.CategoricalAccuracy()])


    print('# Обучаем модель на тестовых данных')
    history = model.fit(X_train, y_train_cat,
                        batch_size=64,
                        epochs=3,
                        validation_data=(X_test, y_test_cat))

    print("Training done, dT:", time.time() - t_start)
    print('\nhistory dict:', history.history)


 # # Оценим модель на тестовых данных, используя "evaluate"
    # print('\n# Оцениваем на тестовых данных')
    # results = model.evaluate(X_test, y_test, batch_size=128)
    # print('test loss, test acc:', results)
    #
    # # Сгенерируем прогнозы (вероятности - выходные данные последнего слоя)
    # # на новых данных с помощью "predict"
    # print('\n# Генерируем прогнозы для 3 образцов')
    # predictions = model.predict(X_test[:3])
    # print('размерность прогнозов:', predictions.shape)
