from BOcnnlstm import BayesianOptimizedCNNLSTM
from BOlstm import BayesianOptimizedLSTM
from BOXGBoost import BayesianOptimizedXGB
from BOrfr import BayesianOptimizedRFR
from BOsvr import BayesianOptimizedSVR

from GwoCeemdan import CEEMDANAnalyzer
from get_data import get_data
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def modelSelection():
    signal = get_data('399300.SZ', '20210101', '20260101',
                      name='CSI300', fields=['close']).values.reshape(-1)
    dates = pd.date_range(start='20210101', periods=len(signal), freq='D')
    # Create analyzer instance
    analyzer = CEEMDANAnalyzer(signal, dates)
    # Define parameter bounds
    lb = [0.01, 10]  # [noise_std lower, ensemble_size lower]
    ub = [0.5, 200]  # [noise_std upper, ensemble_size upper]
    # Optimize and decompose
    imfs = analyzer.optimize_and_decompose(lb, ub, search_agents=10, max_iter=5)
    # Plot results
    analyzer.plot_convergence()
    analyzer.plot_ceemdan_results()
    analyzer.plot_period_analysis(title="CSI300 IMFs Period Analysis", log_scale=True)
    analyzer.plot_period_analysis(title="CSI300 IMFs Period Analysis", log_scale=False)



    cnnlstm = BayesianOptimizedCNNLSTM(n_steps_in=12,batch_size=64,learning_rate=0.001)
    lstm = BayesianOptimizedLSTM(n_steps_in=12,batch_size=64,learning_rate=0.001)
    xgboost=BayesianOptimizedXGB(n_steps_in=12)
    rfr = BayesianOptimizedRFR(n_steps_in=12)
    svr = BayesianOptimizedSVR(n_steps_in=12)
    model=[cnnlstm,lstm,xgboost,rfr,svr]
    data=pd.DataFrame(imfs).T
    # for i in imfs:
    #     for j in model:
    #         x_train,x_val,y_train,y_val=j.prepare_data(data=i)
    model_mse=[]
    model_rmse=[]
    model_mape=[]
    for i in range(data.shape[1]):
        for j in model:
            x_train,x_val,x_test,y_train,y_val,y_test=j.prepare_data(arr=data[[i]])
            print('开始贝叶斯优化...')
            j.bayesian_optimization(x_train,y_train,x_val,y_val,init_points=20,n_iter=8)
            print("\n训练最终模型...")
            history = j.train_final_model(epochs=100)
            # 6. 模型预测
            print("\n进行预测并评估...")
            prediction,mse,rmse,mape = j.predict(x_test, y_test)
            model_mse.append(float(mse))
            model_rmse.append(float(rmse))
            model_mape.append(float(mape))
    print(model_mse)
    print(model_rmse)
    print(model_mape)





