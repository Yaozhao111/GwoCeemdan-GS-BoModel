import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from get_data import get_data
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 同时支持英文字体和中文
matplotlib.rcParams['axes.unicode_minus'] = False

class pre():
    def __init__(self,win_len,hidden_units):
        self.win_len = win_len
        self.hidden_units = hidden_units
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

    def build_model(self,data):
        model_layer = [tf.keras.layers.Reshape((self.win_len, data.shape[1]), input_shape=(self.win_len, data.shape[1])),
                       tf.keras.layers.LSTM(units=self.hidden_units, return_sequences=False),
                       tf.keras.layers.Dense(units=self.hidden_units, activation='sigmoid'),
                       tf.keras.layers.Dense(units=1, activation='sigmoid')]
        model = tf.keras.Sequential(model_layer)
        model.summary()
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

    def train(self,model,x_train,y_train,epochs,batch_size,validation_split):
        model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
        history=model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_split=validation_split)
        loss=history.history['loss']
        print('Loss Curve',loss)

    def predict(self,model,x_test,y_test):
        prediction = model.predict(x_test).reshape(-1)
        prediction = self.scaler_y.inverse_transform(prediction.reshape(-1,1)).reshape(-1)
        y_test = self.scaler_y.inverse_transform(y_test.reshape(-1,1)).reshape(-1)
        pre.plot_result(prediction,y_test)
        print('MSE', (sum(prediction - y_test) ** 2) / len(y_test))
        print('RMSE', np.sqrt((sum(prediction - y_test) ** 2) / len(y_test)))
        print('MAPE (%)', 100 * sum(np.abs(prediction - y_test) / y_test) / len(y_test))

if __name__ == '__main__':
    import lstm_pre
    lstm_pre = lstm_pre.pre(win_len=12,hidden_units=64)
    data,x_train, y_train, x_test, y_test \
        = lstm_pre.data_preparation(label=['close'],
        data=get_data(ts_code='399300.SZ',start_date='2021-01-01',end_date='2026-01-01',
                      fields=['close'],name='CSI300'))
    model=lstm_pre.build_model(data)
    loss=lstm_pre.train(model=model,x_train=x_train,y_train=y_train,epochs=100,batch_size=64,validation_split=0.2)
    lstm_pre.predict(model,x_test,y_test)

