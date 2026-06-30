import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = ['Times New Roman']


class BayesianOptimizedXGB():
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

    def create_model(self, n_estimators, max_depth, learning_rate, min_child_weight,
                     random_state=1412, verbose=False, n_jobs=1):
        reg = xgb.XGBRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            random_state=random_state,
            verbosity=0 if not verbose else 1,
            n_jobs=n_jobs,
            eval_metric='rmse'
        )
        return reg

    def train_model(self, n_estimators, max_depth, learning_rate, min_child_weight,
                    random_state=1412, verbose=False, n_jobs=1):
        n_estimators = int(n_estimators)
        max_depth = int(max_depth)

        model = self.create_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            random_state=1412,
            n_jobs=1,
            verbose=verbose
        )

        # 训练模型
        model.fit(self.x_train, self.y_train,
                  eval_set=[(self.x_val, self.y_val)],
                  verbose=False)

        y_pred = model.predict(self.x_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        return -rmse

    def bayesian_optimization(self, x_train, y_train, x_val, y_val, init_points, n_iter):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        # 定义XGBoost参数搜索空间 - 针对时间序列预测优化
        pbounds = {
            'n_estimators': (50, 300),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.2),
            'min_child_weight': (1, 10)
        }

        self.optimizer = BayesianOptimization(
            f=self.train_model,
            pbounds=pbounds,
            random_state=1412,
            verbose=2
        )

        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        self.best_params = self.optimizer.max['params']

        # 将需要整数的参数转换为整数
        self.best_params['n_estimators'] = int(self.best_params['n_estimators'])
        self.best_params['max_depth'] = int(self.best_params['max_depth'])

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
            n_estimators=self.best_params['n_estimators'],
            max_depth=self.best_params['max_depth'],
            learning_rate=self.best_params['learning_rate'],
            min_child_weight=self.best_params['min_child_weight'],
            n_jobs=-1
        )

        # 训练模型
        self.history = self.best_model.fit(
            self.x_train, self.y_train,
            eval_set=[(self.x_val, self.y_val)],
            verbose=False
        )
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
        plt.title('XGBoost Prediction vs True')
        plt.xlabel('Time Steps')
        # plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.grid(True)
        # plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        if self.best_model is None:
            raise ValueError("没有模型可保存!")
        self.best_model.save_model(filepath)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载已保存的XGBoost模型"""
        self.best_model = xgb.XGBRegressor()
        self.best_model.load_model(filepath)
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
    model = BayesianOptimizedXGB(n_steps_in=12)

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
    # model.save_model("xgboost_model.json")