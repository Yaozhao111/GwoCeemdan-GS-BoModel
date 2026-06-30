# GwoCeemdan-GS-BoModel
设计了一种对股指价格序列进行建模自动化机器学习框架，首先引入灰狼优化算法实现对CEENDAN关键参数的自动寻优，得到一系列本征模态函数；其次使用CNN-LSTM、LSTM、XGBoost、RF以及SVR作为模型选择库，以MSE为目标函数，对各模态函数进行适配性实验，同时在模型拟合时通过贝叶斯优化方法进行超参数调优，最后重构各分量的输出值以实现高精度预测。  
An automatic machine learning framework for modeling stock index price series is designed. First, grey wolf optimization algorithm is introduced to automatically optimize the key parameters of CEENDAN, and a series of intrinsic mode functions are obtained; Secondly, CNN-LSTM, LSTM, XGBoost, RF and SVR are used as the model selection library, MSE is used as the objective function, and adaptability experiments are carried out for each modal function. At the same time, during model fitting, super parameters are optimized by Bayesian optimization method, and finally the output values of each component are reconstructed to achieve high-precision prediction. 
# Methods
总体分为3个阶段。首先获得原始时间序列数据，并输入嵌入GWO的CEMMDAN信号分解方法之中，得到各分量的模态函数，并将其进行标准化处理，此为数据采集与预处理阶段。第二阶段为不同频率的模态函数通过网格搜索法进行模型寻优，确定高频至低频序列的模型使用顺序，使用差异化模型对模态函数进行建模并预测。第三阶段为对比分析，将该方法与融合贝叶斯的单一模型、以及加入CEMMDAN-贝叶斯单一模型进行多次实验，得到其预测精度的分析，分析各类模型预测稳健性与鲁棒性。  
There are three stages. First, obtain the original time series data and input it into the CEMMDAN signal decomposition method embedded in GWO to obtain the modal functions of each component, and standardize them. This is the data acquisition and pre-processing stage. The second stage is to optimize the model of modal functions with different frequencies by grid search method, determine the order of use of the model of high-frequency to low-frequency series, and use the differentiation model to model and predict the modal functions. The third stage is comparative analysis. The method is tested several times with a single Bayesian fusion model and a single Bayesian model added to CEMMDAN to obtain the analysis of its prediction accuracy, and analyze the prediction robustness and robustness of various models.  
          <img width="545" height="464" alt="image" src="https://github.com/user-attachments/assets/4175461f-ad64-4e0b-a31e-008487a6ea68" />

# Environment
numpy==2.1.3  
pandas==2.2.3  
torch==2.10.0  
torchaudio==2.10.0  
torchvision==0.25.0  
tornado==6.5.1  
tqdm==4.67.1  
tensorflow==2.20.0rc0
# Usage
Opening the Main.py, first obtain the model selection sequence through the ModelSelection.py module, and then train and compare various models.
# Dataset
The data from Tushare API (https://tushare.pro), You can use the following Token code: 667d224f69a317ce8609dac2644e05d5fb704224b7288ef2cca5eef7
