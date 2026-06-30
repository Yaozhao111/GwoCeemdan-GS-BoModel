import numpy as np
from get_data import get_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
from bayes_opt import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
plt.rcParams['font.family'] = ['Times New Roman']
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore")


class BayesianOptimizedCNNLSTM():
    def __init__(self, n_steps_in=60, batch_size=64, learning_rate=0.001):
        self.n_steps_in = n_steps_in
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.best_model = None
        self.best_params = None
        self.optimizer = None
        self.history = None

    # def prepare_data(self, data, label):
    #     arrx = self.scaler_x.fit_transform(data)
    #     arry = self.scaler_y.fit_transform(data[label].values.reshape(-1, 1)).reshape(-1)  # 修正reshape
    #     X, y = [], []
    #     for i in range(len(data) - self.n_steps_in):
    #         X.append(arrx[i:(i + self.n_steps_in)])
    #         y.append(arry[i + self.n_steps_in])  # 单步预测
    #     X = np.array(X)
    #     y = np.array(y)
    #     # 70% 训练, 15% 验证, 15% 测试
    #     t = int(0.7 * len(X))
    #     v = int(0.85 * len(X))
    #     x_train, x_val, x_test = X[:t], X[t:v], X[v:]
    #     y_train, y_val, y_test = y[:t], y[t:v], y[v:]
    #     return x_train, x_val, x_test, y_train, y_val, y_test

    def prepare_data(self,arr):
        arr = self.scaler_x.fit_transform(arr).reshape(-1)
        x, y = [], []
        for i in range(len(arr)-self.n_steps_in):
            x.append(arr[i:i+self.n_steps_in])
            y.append(arr[i+self.n_steps_in])
        x=np.array(x)
        y=np.array(y)
        # 70%训练， 15%验证， 15%测试
        t = int(0.7 * len(x))
        v = int(0.85* len(x))
        x_train, x_val, x_test = x[:t], x[t:v], x[v:]
        y_train, y_val, y_test = y[:t], y[t:v], y[v:]
        return x_train, x_val, x_test, y_train, y_val, y_test

    def create_model(self, filters, kernel_size, lstm_units, dense_units, dropout_rate):
        model = Sequential()
        # CNN model
        model.add(Conv1D(filters=int(filters), kernel_size=int(kernel_size), activation='relu',
                         padding='same', input_shape=(self.n_steps_in, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=int(filters), kernel_size=int(kernel_size), activation='relu',
                         padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        # LSTM model
        model.add(LSTM(units=int(lstm_units), return_sequences=True, dropout=dropout_rate))
        model.add(LSTM(units=int(lstm_units), return_sequences=False, dropout=dropout_rate))
        # 全连接层
        model.add(Dense(units=int(dense_units), activation='sigmoid'))
        model.add(Dense(1))  # 单步输出
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def train_model(self, filters, kernel_size, lstm_units, dense_units, dropout_rate):
        filters = int(filters)
        kernel_size = int(kernel_size)
        lstm_units = int(lstm_units)
        dense_units = int(dense_units)
        # 创建模型
        model = self.create_model(filters=filters, kernel_size=kernel_size, lstm_units=lstm_units,
                                  dense_units=dense_units, dropout_rate=dropout_rate)
        # 早停法
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # 训练模型
        history = model.fit(self.X_train, self.y_train, epochs=100, batch_size=self.batch_size,
                            validation_data=(self.X_val, self.y_val), callbacks=[early_stop], verbose=0)
        # 预测验证集
        y_pred = model.predict(self.X_val, verbose=0)
        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        return -rmse  # 返回负RMSE用于最大化

    def bayesian_optimization(self, X_train, y_train, X_val, y_val, init_points, n_iter):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        # 定义参数搜索空间
        pbounds = {
            'filters': (32, 256),  # 卷积核数量
            'kernel_size': (3, 9),  # 卷积核大小
            'lstm_units': (50, 300),  # LSTM单元数
            'dense_units': (10, 200),  # 全连接层单元数
            'dropout_rate': (0, 0)  # Dropout率
        }
        # 创建贝叶斯优化器
        self.optimizer = BayesianOptimization(f=self.train_model, pbounds=pbounds, random_state=42, verbose=2)
        # 执行优化
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        # 获取最佳参数
        self.best_params = self.optimizer.max['params']
        # 将浮点数参数转换为整数
        self.best_params['filters'] = int(self.best_params['filters'])
        self.best_params['kernel_size'] = int(self.best_params['kernel_size'])
        self.best_params['lstm_units'] = int(self.best_params['lstm_units'])
        self.best_params['dense_units'] = int(self.best_params['dense_units'])
        print("\n贝叶斯优化完成!")
        print(f"最佳验证RMSE: {-self.optimizer.max['target']:.6f}")
        print("最佳超参数组合:")
        for k, v in self.best_params.items():
            print(f"{k}: {v}")

    def train_final_model(self, epochs=200):
        if self.best_params is None:
            raise ValueError("请先执行贝叶斯优化!")
        # 创建最终模型
        self.best_model = self.create_model(
            filters=self.best_params['filters'],
            kernel_size=self.best_params['kernel_size'],
            lstm_units=self.best_params['lstm_units'],
            dense_units=self.best_params['dense_units'],
            dropout_rate=self.best_params['dropout_rate']
        )
        # 训练模型
        self.history = self.best_model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=self.batch_size,
                                           validation_data=(self.X_val, self.y_val), verbose=1)
        print("最终模型训练完成!")
        return self.best_model

    # 添加正确的预测方法
    def predict(self, x_test, y_test):
        if self.best_model is None:
            raise ValueError("请先训练最终模型!")

        # 进行预测
        prediction = self.best_model.predict(x_test).flatten()

        # 反标准化
        prediction = self.scaler_x.inverse_transform(prediction.reshape(-1, 1)).flatten()
        y_test_orig = self.scaler_x.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # 绘制结果
        self.plot_result(prediction, y_test_orig)

        # 计算指标
        mse = mean_squared_error(y_test_orig, prediction)
        rmse = np.sqrt(mse)
        mape = 100 * np.mean(np.abs((y_test_orig - prediction) / y_test_orig))
        r2=r2_score(y_test_orig, prediction)

        print(f'MSE: {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAPE: {mape:.2f}%')
        print(f'R2: {r2:.4f}')

        return prediction,mse,rmse,mape,r2

    # 修正为实例方法
    def plot_result(self, prediction, y_test):
        plt.figure(figsize=(3, 2))
        plt.plot(y_test, 'b-', label='True')
        plt.plot(prediction, 'r--', label='Pred.')
        plt.title('CNN-LSTM Prediction vs True')
        plt.xlabel('Time Steps')
        # plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

    def plot_training_history(self):
        if self.history is None:
            print("没有训练历史可显示!")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='训练损失')
        plt.plot(self.history.history['val_loss'], label='验证损失')
        plt.title('模型训练过程')
        plt.ylabel('MSE损失')
        plt.xlabel('训练轮次')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, filepath):
        if self.best_model is None:
            raise ValueError("没有模型可保存!")
        self.best_model.save(filepath)
        print(f"模型已保存到 {filepath}")


# 使用示例
if __name__ == "__main__":
    # 1. 创建示例数据
    data = get_data(ts_code='399300.SZ', start_date='20210101', end_date='20260101',
                    fields=['close'], name='CSI300')
    print(data)
    # 2. 初始化模型
    model = BayesianOptimizedCNNLSTM(n_steps_in=12, batch_size=64, learning_rate=0.001)
    # 3. 数据预处理
    x_train, x_val, x_test, y_train, y_val, y_test = model.prepare_data(arr=data)
    # 4. 执行贝叶斯优化
    print("开始贝叶斯优化...")
    model.bayesian_optimization(x_train, y_train, x_val, y_val, init_points=5, n_iter=200)
    # 5. 使用最佳参数训练最终模型
    print("\n训练最终模型...")
    history = model.train_final_model(epochs=100)
    # 6. 模型预测
    print("\n进行预测并评估...")
    prediction,mse,rmse,mape = model.predict(x_test, y_test)
    model.plot_training_history()