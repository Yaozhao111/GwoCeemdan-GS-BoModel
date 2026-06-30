import numpy as np
import pandas as pd
from get_data import get_data
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
#优化器
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = ['Times New Roman']

class BayesianOptimizedRFR():
    def __init__(self,n_steps_in=60):
        self.n_steps_in=n_steps_in
        self.scaler_x=MinMaxScaler()
        self.scaler_y=MinMaxScaler()
        self.best_model=None
        self.best_params=None
        self.optimizer=None
        self.history=None

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


    def create_model(self,n_estimators,max_depth,max_features,min_impurity_decrease,random_state=1412,
                     verbose=False,n_jobs=1):
        reg=RFR(n_estimators=int(n_estimators),max_depth=int(max_depth),max_features=int(max_features),
                min_impurity_decrease=min_impurity_decrease,random_state=1412,verbose=verbose,n_jobs=-1)
        return reg

    def train_model(self,n_estimators,max_depth,max_features,min_impurity_decrease,random_state=1412,
                    verbose=False,n_jobs=1):
        n_estimators=int(n_estimators)
        max_depth=int(max_depth)
        max_features=int(max_features)
        min_impurity_decrease=min_impurity_decrease
        model=self.create_model(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,
                                min_impurity_decrease=min_impurity_decrease,random_state=1412)
        history=model.fit(self.x_train,self.y_train)
        y_pred=model.predict(self.x_val)
        rmse=np.sqrt(mean_squared_error(self.y_val,y_pred))
        return -rmse

    def bayesian_optimization(self,x_train,y_train,x_val,y_val,init_points,n_iter):
        self.x_train=x_train
        self.y_train=y_train
        self.x_val=x_val
        self.y_val=y_val
        # 定义参数搜索空间
        pbounds={
            'n_estimators':(80,100),
            'max_depth':(20,100),
            'max_features':(10,20),
            'min_impurity_decrease':(0,1)
        }
        self.optimizer=BayesianOptimization(f=self.train_model,pbounds=pbounds,random_state=1412,verbose=2)
        self.optimizer.maximize(init_points=init_points,n_iter=n_iter)
        self.best_params=self.optimizer.max['params']
        # 将浮点数参数转换为整数
        self.best_params['n_estimators']=int(self.best_params['n_estimators'])
        self.best_params['max_depth']=int(self.best_params['max_depth'])
        self.best_params['max_features']=int(self.best_params['max_features'])
        print('贝叶斯优化完成！')
        print(f'最佳验证RMSE：{-self.optimizer.max['target']:.6f}')
        print('最佳参数组合：')
        for k,v in self.best_params.items():
            print(f'{k}: {v}')

    def train_final_model(self,epochs):
        if self.best_params is None:
            raise ValueError("请先执行贝叶斯优化!")
        # 创建最终模型
        self.best_model = self.create_model(
            n_estimators=self.best_params['n_estimators'],
            max_depth=self.best_params['max_depth'],
            max_features=self.best_params['max_features'],
            min_impurity_decrease=self.best_params['min_impurity_decrease'],
        )
        # 训练模型
        self.history = self.best_model.fit(self.x_train, self.y_train)
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
        print(f'MAPE: {mape:.4f}%')
        print(f'R2: {r2:.4f}')

        return prediction,mse,rmse,mape,r2

    # 修正为实例方法
    def plot_result(self, prediction, y_test):
        plt.figure(figsize=(3, 2))
        plt.plot(y_test, 'b-', label='True')
        plt.plot(prediction, 'r--', label='Pred.')
        plt.title('RFR Prediction vs True')
        plt.xlabel('Time Steps')
        # plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

    # def plot_training_history(self):
    #     if self.history is None:
    #         print("没有训练历史可显示!")
    #         return
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(self.history.history['loss'], label='训练损失')
    #     plt.plot(self.history.history['val_loss'], label='验证损失')
    #     plt.title('模型训练过程')
    #     plt.ylabel('MSE损失')
    #     plt.xlabel('训练轮次')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

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
    model = BayesianOptimizedRFR(n_steps_in=12)
    # 3. 数据预处理
    x_train, x_val, x_test, y_train, y_val, y_test = model.prepare_data(arr=data)
    # 4. 执行贝叶斯优化
    print("开始贝叶斯优化...")
    print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)
    model.bayesian_optimization(x_train, y_train, x_val, y_val, init_points=20, n_iter=7)
    # 5. 使用最佳参数训练最终模型
    print("\n训练最终模型...")
    history = model.train_final_model(epochs=100)
    # 6. 模型预测
    print("\n进行预测并评估...")
    prediction,mse,rmse,mape,r2 = model.predict(x_test, y_test)
    # model.plot_training_history()
