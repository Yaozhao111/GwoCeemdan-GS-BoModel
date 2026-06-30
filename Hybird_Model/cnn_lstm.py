import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from get_data import get_data
import matplotlib
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional,Conv1D, MaxPooling1D,TimeDistributed,Flatten
import warnings
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 同时支持英文字体和中文
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class pre():
    def __init__(self,win_len,features,filters,kernal_size,hidden_units,Dropout):
        self.win_len = win_len
        self.features = features
        self.filters = filters
        self.kernal_size = kernal_size
        self.input_shape = (win_len,1)
        self.hidden_units = hidden_units
        self.Dropout = Dropout
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def data_preparation(self,data,label):
        arrx = self.scaler_x.fit_transform(data)
        arry = self.scaler_y.fit_transform(data[label]).reshape(-1)
        x = [];y = []
        for i in range(len(arrx) - self.win_len - 1):
            x.append(arrx[i:i + self.win_len])
            y.append(arry[i + self.win_len])
        x = np.array(x)
        y = np.array(y)
        t = int(0.8 * len(x))
        x_train = x[:t]
        x_test = x[t:]
        y_train = y[:t]
        y_test = y[t:]
        print(data.shape)
        return data,x_train, y_train, x_test, y_test

    def build_model(self):
        # Part 1: CNN model
        model = tf.keras.Sequential()
        model.add(Conv1D(filters=self.filters,kernel_size=self.kernal_size,input_shape=self.input_shape,
                         padding='same',activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=self.filters,kernel_size=self.kernal_size,input_shape=self.input_shape,
                         padding='same',activation='relu'))
        model.add(Dropout(rate=self.Dropout))
        # Part 2: LSTM model
        model.add(LSTM(units=self.hidden_units,return_sequences=True,dropout=self.Dropout))
        model.add(LSTM(units=self.hidden_units, return_sequences=False, dropout=self.Dropout))
        model.add(Dense(units=self.hidden_units,activation='sigmoid'))
        model.add(Dense(units=1,activation='sigmoid'))

        # model_layer = [tf.keras.layers.Reshape((self.win_len, data.shape[1]), input_shape=(self.win_len, data.shape[1])),
        #                tf.keras.layers.LSTM(units=self.hidden_units, return_sequences=False),
        #                tf.keras.layers.Dense(units=self.hidden_units, activation='sigmoid'),
        #                tf.keras.layers.Dense(units=1, activation='sigmoid')]
        # model = tf.keras.Sequential(model_layer)
        # model.summary()

        return model

    def plot_loss(loss):
        fig=plt.figure(figsize=(5,3))
        ax=fig.add_subplot(111)
        ax.plot(range(len(loss)),loss)
        plt.grid(True)
        plt.show()

    def plot_result(prediction,true):
        fig=plt.figure(figsize=(5,3))
        ax=fig.add_subplot(111)
        ax.plot(range(len(prediction)),prediction)
        ax.plot(range(len(prediction)),true)
        plt.grid(True)
        ax.legend(['Prediction','Real'])
        plt.show()

    def fit(self,model,x_train,y_train,epochs,batch_size,validation_split):
        model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
        history=model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_split=validation_split)
        loss=history.history['loss']
        print('Loss Curve',loss)
        pre.plot_loss(loss=loss)

    def predict(self,model,x_test,y_test):
        prediction = model.predict(x_test).reshape(-1)
        prediction = self.scaler_y.inverse_transform(prediction.reshape(-1,1)).reshape(-1)
        y_test = self.scaler_y.inverse_transform(y_test.reshape(-1,1)).reshape(-1)
        pre.plot_result(prediction,y_test)
        print('MSE', (sum(prediction - y_test) ** 2) / len(y_test))
        print('RMSE', np.sqrt((sum(prediction - y_test) ** 2) / len(y_test)))
        print('MAPE (%)', 100 * sum(np.abs(prediction - y_test) / y_test) / len(y_test))

if __name__ == '__main__':
    import cnn_lstm
    cnn_lstm_pre = cnn_lstm.pre(win_len=32,features=1,filters=32,kernal_size=2,hidden_units=128,Dropout=0)
    data,x_train, y_train, x_test, y_test \
        = cnn_lstm_pre.data_preparation(label=['close'],
        data=get_data(ts_code='399300.SZ',start_date='2021-01-01',end_date='2026-01-01',
                      fields=['close'],name='CSI300'))
    model=cnn_lstm_pre.build_model(data)
    loss=cnn_lstm_pre.fit(model=model,x_train=x_train,y_train=y_train,epochs=100,batch_size=64,validation_split=0.2)
    cnn_lstm_pre.predict(model,x_test,y_test)

