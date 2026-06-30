import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = ['Times New Roman']


class BayesianOptimizedSVR():
    def __init__(self, n_steps_in=60):
        self.n_steps_in = n_steps_in
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.best_model = None
        self.best_params = None
        self.optimizer = None
        self.history = None

    def prepare_data(self, arr):
        arr = self.scaler_x.fit_transform(arr).reshape(-1)
        x, y = [], []
        for i in range(len(arr) - self.n_steps_in):
            x.append(arr[i:i + self.n_steps_in])
            y.append(arr[i + self.n_steps_in])
        x = np.array(x)
        y = np.array(y)
        # 70%训练，15%验证，15%测试
        t = int(0.7 * len(x))
        v = int(0.85 * len(x))
        x_train, x_val, x_test = x[:t], x[t:v], x[v:]
        y_train, y_val, y_test = y[:t], y[t:v], y[v:]
        return x_train, x_val, x_test, y_train, y_val, y_test

    def create_model(self, C, epsilon, gamma, kernel='rbf',
                     random_state=1412, verbose=False):
        reg = SVR(
            C=C,
            epsilon=epsilon,
            gamma=gamma,
            kernel=kernel,
            cache_size=500,  # 提高缓存大小以加速计算
            max_iter=-1  # 不限制迭代次数
        )
        return reg

    def train_model(self, C, epsilon, gamma,
                    random_state=1412, verbose=False):
        model = self.create_model(
            C=C,
            epsilon=epsilon,
            gamma=gamma,
            kernel='rbf',
            random_state=1412
        )

        # 训练模型
        history = model.fit(self.x_train, self.y_train)

        y_pred = model.predict(self.x_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        return -rmse

    def bayesian_optimization(self, x_train, y_train, x_val, y_val, init_points, n_iter):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        # 定义SVR参数搜索空间 - 针对时间序列预测优化
        pbounds = {
            'C': (0.1, 100),
            'epsilon': (0.01, 1.0),
            'gamma': (0.001, 1.0)
        }

        self.optimizer = BayesianOptimization(
            f=self.train_model,
            pbounds=pbounds,
            random_state=1412,
            verbose=2
        )

        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        self.best_params = self.optimizer.max['params']

        print('贝叶斯优化完成！')
        print(f'最佳验证RMSE：{-self.optimizer.max["target"]:.6f}')
        print('最佳参数组合：')
        for k, v in self.best_params.items():
            print(f'{k}: {v}')

        return self.best_params

    def train_final_model(self, epochs=None):
        if self.best_params is None:
            raise ValueError("请先执行贝叶斯优化!")

        # 创建最终模型
        self.best_model = self.create_model(
            C=self.best_params['C'],
            epsilon=self.best_params['epsilon'],
            gamma=self.best_params['gamma'],
            kernel='rbf'
        )

        # 训练模型
        self.history = self.best_model.fit(self.x_train, self.y_train)
        print("最终模型训练完成!")
        return self.best_model

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
        print(f'MAPE: {mape:.4f}%')
        print(f'R2: {r2:.4f}')

        return prediction, mse, rmse, mape,r2

    def plot_result(self, prediction, y_test):
        plt.figure(figsize=(3, 2))
        plt.plot(y_test, 'b-', label='True')
        plt.plot(prediction, 'r--', label='Pred.')
        plt.title('SVR Prediction vs True')
        plt.xlabel('Time Steps')
        # plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.grid(True)
        # plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        if self.best_model is None:
            raise ValueError("没有模型可保存!")

        # 保存模型参数
        import joblib
        model_data = {
            'model': self.best_model,
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'best_params': self.best_params,
            'n_steps_in': self.n_steps_in
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载已保存的SVR模型"""
        import joblib
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.scaler_x = model_data['scaler_x']
        self.scaler_y = model_data['scaler_y']
        self.best_params = model_data['best_params']
        self.n_steps_in = model_data['n_steps_in']
        print(f"模型已从 {filepath} 加载")
        return self.best_model


# 使用示例
if __name__ == "__main__":
    # 1. 创建示例数据（使用模拟数据，因为get_data函数可能不可用）
    np.random.seed(1412)
    n_samples = 1000
    # 创建时间序列数据
    time = np.arange(n_samples)
    data = 100 + np.sin(time * 0.05) * 30 + time * 0.1 + np.random.randn(n_samples) * 5
    data = data.reshape(-1, 1)

    print(f"数据形状: {data.shape}")
    print(f"数据示例（前5个）: {data[:5].flatten()}")

    # 2. 初始化模型
    model = BayesianOptimizedSVR(n_steps_in=12)

    # 3. 数据预处理
    x_train, x_val, x_test, y_train, y_val, y_test = model.prepare_data(arr=data)

    print(f"训练数据形状: x_train{x_train.shape}, y_train{y_train.shape}")
    print(f"验证数据形状: x_val{x_val.shape}, y_val{y_val.shape}")
    print(f"测试数据形状: x_test{x_test.shape}, y_test{y_test.shape}")

    # 4. 执行贝叶斯优化
    print("\n开始贝叶斯优化...")
    best_params = model.bayesian_optimization(
        x_train, y_train, x_val, y_val,
        init_points=10, n_iter=5
    )

    # 5. 使用最佳参数训练最终模型
    print("\n训练最终模型...")
    history = model.train_final_model()

    # 6. 模型预测
    print("\n进行预测并评估...")
    prediction, mse, rmse, mape = model.predict(x_test, y_test)

    # 7. 保存模型（可选）
    # model.save_model("svr_model.pkl")