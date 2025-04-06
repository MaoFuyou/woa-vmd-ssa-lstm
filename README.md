# WOA-VMD-SSA-LSTM 项目

## 一、项目概述
本项目主要围绕时间序列预测展开，综合运用了鲸鱼优化算法（Whale Optimization Algorithm, WOA）、变分模态分解（Variational Mode Decomposition, VMD）、麻雀搜索算法（Sparrow Search Algorithm, SSA）和长短期记忆网络（Long Short-Term Memory, LSTM）。旨在通过多种算法的结合，提高时间序列预测的准确性和稳定性。

## 二、项目结构
项目的主要文件及功能如下：
- `calc_error.m`：用于计算预测误差指标，如均方根误差（RMSE）、平均绝对误差（MAE）和平均绝对百分比误差（MAPE）等。
- `CNN.m`：构建并训练卷积神经网络（CNN）进行时间序列预测，包含数据处理、模型训练、预测和评估等步骤。
- `fun.m`：定义适应度函数，用于评估 LSTM 模型的性能，同时进行模型训练。
- `data_process.m`：对输入的时间序列数据进行处理，生成训练集和测试集。
- `SVR.m`：构建并训练支持向量回归（SVR）模型进行时间序列预测，包含数据处理、模型训练、预测和评估等步骤。
- `initialization.m`：初始化搜索代理的位置，为优化算法提供初始种群。
- `vmdtest.m`：进行 VMD 分解测试，对输入信号进行分解并可视化分解结果。
- `WOA.m`：实现鲸鱼优化算法，用于优化模型的参数。
- `VMD.m`：实现变分模态分解算法，将输入信号分解为多个模态。
- `SSA.m`：实现麻雀搜索算法，用于优化 LSTM 模型的参数。
- `WOAVMD.m`：结合 WOA 和 VMD，使用 WOA 优化 VMD 的参数。
- `VMD_SSA_LSTM.m`：整合 VMD、SSA 和 LSTM 进行时间序列预测，包括单独的 LSTM 预测、VMD-LSTM 预测和 VMD-SSA-LSTM 预测，并对比不同模型的预测效果。

## 三、依赖库
本项目主要使用 MATLAB 实现，需要以下 MATLAB 工具箱：
- Neural Network Toolbox：用于构建和训练神经网络模型。
- Statistics and Machine Learning Toolbox：用于数据处理和模型评估。

## 四、使用方法

### 1. 数据准备
将时间序列数据保存为 Excel 文件，例如 `Data1.xlsx`、`Data2.xlsx`、`Data3.xlsx` 等，并确保文件路径和数据范围在代码中正确设置。

### 2. 运行代码
- **单独 LSTM 预测**：运行 `VMD_SSA_LSTM.m` 文件中的单独 LSTM 预测部分，代码会自动进行数据处理、模型训练、预测和评估，并输出训练集和测试集的误差指标。
- **VMD-LSTM 预测**：首先运行 `vmdtest.m` 进行 VMD 分解，将分解结果保存为 `vmd_data.mat` 文件。然后运行 `VMD_SSA_LSTM.m` 文件中的 VMD-LSTM 预测部分，代码会对每个分解模态分别进行 LSTM 预测，并将预测结果累加，最后输出训练集和测试集的误差指标。
- **VMD-SSA-LSTM 预测**：在完成 VMD 分解后，运行 `VMD_SSA_LSTM.m` 文件中的 VMD-SSA-LSTM 预测部分，代码会使用 SSA 优化 LSTM 模型的参数，然后对每个分解模态分别进行 LSTM 预测，并将预测结果累加，最后输出训练集和测试集的误差指标。

### 3. 参数调整
你可以根据需要调整代码中的参数，例如：
- `pop`：种群数量，用于优化算法。
- `Max_iter`：最大迭代次数，用于优化算法。
- `lb` 和 `ub`：参数的上下界，用于优化算法。
- `numHiddenUnits`：LSTM 层的隐藏单元数量。
- `MaxEpochs`：最大训练轮数。
- `InitialLearnRate`：初始学习率。

## 五、实验结果
运行代码后，会在命令窗口输出不同模型的训练集和测试集的误差指标，包括 RMSE、MAE 和 MAPE 等。同时，还会生成相应的可视化图表，展示模型的训练过程和预测结果。

## 六、注意事项
- 确保 MATLAB 环境中安装了所需的工具箱。
- 数据文件的路径和数据范围需要根据实际情况进行调整。
- 优化算法的参数需要根据具体问题进行调整，以获得更好的优化效果。

## 七、贡献者
[Fuyou Mao]


