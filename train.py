import tensorflow as tf
import pandas as pd
import numpy as np
class Module():


    def __init__(self):
        pass

    def NN_model(self):
        # Input:784 --> Hidden:256 --> Hidden:64 --> Hidden:10 --> Softmax --> Output
        # Loss Func: CrossEntropy  /  Optimizer: Adam Optimizer
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def CNN_model(self):
        # Input:28x28x1 --> Conv:26x26x16 --> Conv:24x24x32 --> Pooling:12x12x32 --> Hidden:256 --> Hidden:64 --> Hidden:10 --> Softmax --> Output
        # Loss Func: CrossEntropy  /  Optimizer: Adam Optimizer
        # Etc: BatchNormalization
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, model, train, label, epoch, batch_size):
        hist = model.fit(train, label, epochs=epoch, batch_size=batch_size)
        return hist

    def print_train_result(self, train):
        print('## train loss and acc ##')
        print(train.history['loss'])
        print(train.history['acc'])

    def model_predict(self, model, X_test):
        return model.predict(X_test)

    def argument_maximize(self, prediction):
        result = []
        for i, predict in enumerate(prediction):
            argmax = np.argmax(predict)
            result.append([i+1, argmax])
        return result

    def model_evaluate(self, model, X_test, label, batch_size):
        return model.evaluate(X_test, label, batch_size=batch_size)

    def save_csv(self, result, filename):
        df = pd.DataFrame(data=np.array(result), columns=['ImageId', 'Label'])
        df.to_csv(filename+".csv", header=True, index=False)

#   Data Load and Save
data = pd.read_csv('./all/train.csv', sep=",", dtype='unicode')   # Data Load
test = pd.read_csv('./all/test.csv', sep=",", dtype='unicode')   # Data Load
train_label = data["label"] # Train Data의 Label 값 저장
train_data = data[data.columns[1:]] # Train Data의 값들 저장
train_label_data = np.array(train_label.values) # array 형식으로 저장
train_label_data = tf.keras.utils.to_categorical(train_label_data, 10)
train_data_data = np.array(train_data.values)   # array 형식으로 저장
X_train = train_data_data.reshape(train_data_data.shape[0], 28, 28, 1).astype('float32')
test_data = np.array(test.values)
X_test = test_data.reshape(test_data.shape[0], 28, 28, 1).astype('float32')
'''
#   Neural Network Train and Evaluate
module1 = Module()
model = module1.NN_model()
train = module1.train(model=model, train=train_data_data, label= train_label_data, epoch=30, batch_size=1000)
module1.print_train_result(train)
predictions = module1.model_predict(model, test_data)
result = module1.argument_maximize(predictions)
module1.save_csv(result, 'NN_result')
'''
#   Convolutional Neural Network Train and Evaluate
module2 = Module()
model = module2.CNN_model()
train = module2.train(model=model, train = X_train, label=train_label_data, epoch=30, batch_size=1000)
module2.print_train_result(train)
predictions = module2.model_predict(model, X_test)
result = module2.argument_maximize(predictions)
module2.save_csv(result, 'CNN_result')
