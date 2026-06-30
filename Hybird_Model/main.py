from BOcnnlstm import BayesianOptimizedCNNLSTM
from BOlstm import BayesianOptimizedLSTM
from BOXGBoost import BayesianOptimizedXGB
from BOrfr import BayesianOptimizedRFR
from BOsvr import BayesianOptimizedSVR
from GwoCeemdan import CEEMDANAnalyzer
from get_data import get_data
from modelSelection import modelSelection
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

modelSelection()

signal = get_data('399300.SZ', '20210701', '20260701',
                  name='CSI300', fields=['close']).values.reshape(-1)
dates = pd.date_range(start='20250101', periods=len(signal), freq='D')
# Create analyzer instance
analyzer = CEEMDANAnalyzer(signal, dates)
# Define parameter bounds
lb = [0.01, 10]  # [noise_std lower, ensemble_size lower]
ub = [0.5, 200]  # [noise_std upper, ensemble_size upper]
# Optimize and decompose
imfs = analyzer.optimize_and_decompose(lb, ub, search_agents=10, max_iter=5)
data=pd.DataFrame(imfs).T
# Plot results
analyzer.plot_convergence()
analyzer.plot_ceemdan_results()

cnnlstm = BayesianOptimizedCNNLSTM(n_steps_in=12,batch_size=64,learning_rate=0.001)
lstm = BayesianOptimizedLSTM(n_steps_in=12,batch_size=64,learning_rate=0.001)
xgboost=BayesianOptimizedXGB(n_steps_in=12)
rfr = BayesianOptimizedRFR(n_steps_in=12)
svr = BayesianOptimizedSVR(n_steps_in=12)

model_order=pd.Series([cnnlstm,lstm,xgboost,svr,rfr,svr,svr])
imf_pre=[]
#Modeling Imfi via different model
for i in range(len(model_order)):
    x_train,x_val,x_test,y_train,y_val,y_test=model_order[i].prepare_data(arr=data[[i]])
    model_order[i].bayesian_optimization(x_train, y_train, x_val, y_val, init_points=20, n_iter=8)
    print("\n训练最终模型...")
    history = model_order[i].train_final_model(epochs=100)
    # 6. 模型预测
    print("\n进行预测并评估...")
    prediction, mse, rmse, mape,r2 = model_order[i].predict(x_test, y_test)
    imf_pre.append(prediction)
# print(imf_pre)
imf_pre = pd.DataFrame(imf_pre)
construct=imf_pre.sum().values
fig=plt.figure(figsize=(5,2))
plt.plot(construct,'r--',label='Pred.')
plt.plot(signal[-len(construct):],'b-',label='True')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
mse=mean_squared_error(signal[-len(construct):], construct)
rmse=np.sqrt(mse)
mape=100*mean_absolute_percentage_error(signal[-len(construct):], construct)
r2=r2_score(signal[-len(construct):], construct)
print('The MSE,RMSE,MAPE and R2 of hybird model is:',mse,rmse,mape,r2)
# Modeling with single without CEEMDAN
model_set=[cnnlstm,lstm,xgboost,rfr,svr]
single_pre=[]
for i in range(len(model_set)):
    x_train,x_val,x_test,y_train,y_val,y_test=model_set[i].prepare_data(arr=signal.reshape(-1,1))
    model_set[i].bayesian_optimization(x_train, y_train, x_val, y_val, init_points=20, n_iter=8)
    print("\n训练最终模型...")
    history = model_set[i].train_final_model(epochs=100)
    # 6. 模型预测
    print("\n进行预测并评估...")
    prediction, mse, rmse, mape,r2 = model_set[i].predict(x_test, y_test)

    single_pre.append(prediction)
    print('The mse, rmse, mape and R2 of Model \n{} is {}, {}, {}, {}'.format(
        model_set[i], mse, rmse, mape,r2))

#CEEMDAN-Single model
for i in range(len(model_order)):
    ceemd_single_pre = []
    for j in range(len(model_order)):
        x_train,x_val,x_test,y_train,y_val,y_test=model_order[i].prepare_data(arr=data[[j]])
        model_order[i].bayesian_optimization(x_train, y_train, x_val, y_val, init_points=20, n_iter=8)
        print("\n训练最终模型...")
        history = model_order[i].train_final_model(epochs=100)
        # 6. 模型预测
        print("\n进行预测并评估...")
        prediction, mse, rmse, mape,r2 = model_order[i].predict(x_test, y_test)
        ceemd_single_pre.append(prediction)

    ceemd_single_pre=pd.DataFrame(ceemd_single_pre)
    construct=ceemd_single_pre.sum().values
    mse=mean_squared_error(signal[-len(construct):], construct)
    rmse=np.sqrt(mse)
    mape=mean_absolute_percentage_error(signal[-len(construct):], construct)*100
    r2=r2_score(signal[-len(construct):], construct)
    print('The MSE,RMSE,MAPE and R2 of Ceendam-{} is \n {}, {}, {}, {}'.format(
        model_order[i], mse, rmse, mape,r2))
    plt.figure(figsize=(3,2))
    plt.plot(construct,'r--',label='Pred.')
    plt.plot(signal[-len(construct):],'b-',label='True')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title('Si.Cee pred. for ceem.-{}'.format(model_order[i]))
    plt.show()