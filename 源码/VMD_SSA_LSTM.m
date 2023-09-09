clc;
clear 
close all

%% 单一的LSTM预测
tic
disp('…………………………………………………………………………………………………………………………')
disp('单一的LSTM预测')
disp('…………………………………………………………………………………………………………………………')
data = xlsread('Data1.xlsx','D3:D357');
[x,y]=data_process(data,24);%前24个时刻 预测下一个时刻
%归一化
[xs,mappingx]=mapminmax(x',0,1);x=xs';
[ys,mappingy]=mapminmax(y',0,1);y=ys';
%划分数据
n=size(x,1);
m=round(n*0.7);%前70%训练，对最后30%进行预测

%归一化后的训练和测试集划分
xTrain=x(1:m,:)';
xTest=x(m+1:end,:)';
yTrain=y(1:m,:)';
yTest=y(m+1:end,:)';

%% 创建LSTM回归网络，序列预测，因此，输入24维，输出一维
numFeatures = size(xTrain,1);
numResponses = 1;

%% 基础LSTM测试
numHiddenUnits = 35;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
 
%指定训练选项
options = trainingOptions('adam', ...
    'MaxEpochs',35, ...
    'ExecutionEnvironment' ,'cpu',...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'LearnRateDropFactor',0.2, ...%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率
    'L2Regularization',0.0001,...
    'Verbose',0);
%训练LSTM
net = trainNetwork(xTrain,yTrain,layers,options);


%训练集测试
numTimeStepsTrain = size(xTrain,2);
for i = 1:numTimeStepsTrain
    [net,predictTrain(:,i)] = predictAndUpdateState(net,xTrain(:,i),'ExecutionEnvironment','cpu');
end

%测试集测试
numTimeStepsTest = size(xTest,2);
for i = 1:numTimeStepsTest
    [net,predictTest(:,i)] = predictAndUpdateState(net,xTest(:,i),'ExecutionEnvironment','cpu');
    %参数更新太多，不能用predict，用predictAndUpdateState
end


% 反归一化
PredictTrain=mapminmax('reverse',predictTrain,mappingy);
PredictTest=mapminmax('reverse',predictTest,mappingy);
YTrain=mapminmax('reverse',yTrain,mappingy);
YTest=mapminmax('reverse',yTest,mappingy);

disp('')
disp('训练集误差指标:')
[mae1,rmse1,mape1,error1]=calc_error(YTrain,PredictTrain);
fprintf('\n')

disp('测试集误差指标:')
[mae2,rmse2,mape2,error2]=calc_error(YTest,PredictTest);
fprintf('\n')
toc


%% VMD-LSTM预测

tic
disp('…………………………………………………………………………………………………………………………')
disp('VMD-LSTM预测')
disp('…………………………………………………………………………………………………………………………')
load vmd_data
imf=u;
c=size(imf,1);
%% 对每个分量建模
for d=1:c
disp(['第',num2str(d),'个分量建模'])    
data_imf=imf(d,:);
[x_imf,y_imf]=data_process(data_imf,24);%前24个时刻 预测下一个时刻
%归一化
[xs_imf,mappingx_imf]=mapminmax(x_imf',0,1);x_imf=xs_imf';
[ys_imf,mappingy_imf]=mapminmax(y_imf',0,1);y_imf=ys_imf';
%划分数据
n=size(x_imf,1);
m=round(n*0.7);%前70%训练，对最后30%进行预测

%归一化后的训练和测试集划分
xTrain_imf=x_imf(1:m,:)';
xTest_imf=x_imf(m+1:end,:)';
yTrain_imf=y_imf(1:m,:)';
yTest_imf=y_imf(m+1:end,:)';

%% 创建LSTM回归网络，序列预测，因此，输入24维，输出一维
numFeatures = size(xTrain_imf,1);
numResponses = 1;

%% 基础LSTM测试
numHiddenUnits = 35;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
 
%指定训练选项
options = trainingOptions('adam', ...
    'MaxEpochs',35, ...
    'ExecutionEnvironment' ,'cpu',...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'LearnRateDropFactor',0.2, ...%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率
    'L2Regularization',0.0001,...
    'Verbose',0);
%训练LSTM
net = trainNetwork(xTrain_imf,yTrain_imf,layers,options);


%训练集测试
numTimeStepsTrain = size(xTrain_imf,2);
for i = 1:numTimeStepsTrain
    [net,predictTrain_imf(:,i)] = predictAndUpdateState(net,xTrain_imf(:,i),'ExecutionEnvironment','cpu');
end

%测试集测试
numTimeStepsTest = size(xTest_imf,2);
for i = 1:numTimeStepsTest
    [net,predictTest_imf(:,i)] = predictAndUpdateState(net,xTest_imf(:,i),'ExecutionEnvironment','cpu');
    %参数更新太多，不能用predict，用predictAndUpdateState
end


% 反归一化
PredictTrain_imf(d,:)=mapminmax('reverse',predictTrain_imf,mappingy_imf);
PredictTest_imf(d,:)=mapminmax('reverse',predictTest_imf,mappingy_imf);
YTrain_imf(d,:)=mapminmax('reverse',yTrain_imf,mappingy_imf);
YTest_imf(d,:)=mapminmax('reverse',yTest_imf,mappingy_imf);

end
% 各分量预测的结果相加
PredictTrain_imf=sum(PredictTrain_imf);
PredictTest_imf=sum(PredictTest_imf);
YTrain_imf=sum(YTrain_imf);
YTest_imf=sum(YTest_imf);


disp('')
disp('训练集误差指标:')
[mae3,rmse3,mape3,error3]=calc_error(YTrain_imf,PredictTrain_imf);
fprintf('\n')

disp('测试集误差指标:')
[mae4,rmse4,mape4,error4]=calc_error(YTest_imf,PredictTest_imf);
fprintf('\n')
toc;



%% VMD-SSA-LSTM预测
tic
disp('…………………………………………………………………………………………………………………………')
disp('VMD-SSA-LSTM预测')
disp('…………………………………………………………………………………………………………………………')
pop=10; % 麻雀数量
Max_iteration=20; % 最大迭代次数
dim=3; % 优化lstm的3个参数
lb = [40,40,0.001];%下边界
ub = [200,200,0.03];%上边界
fobj = @(x) fun(x,numFeatures,numResponses,xTrain,yTrain,xTest,yTest);
%基础麻雀算法
[Best_pos,Best_score,SSA_curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj); %开始优化

%% 绘制进化曲线
figure
plot(SSA_curve,'r-','linewidth',3)
xlabel('进化代数')
ylabel('均方误差MSE')
legend('最佳适应度')
title('SSA-LSTM的进化收敛曲线')

disp('')
disp(['最优隐藏单元数目为   ',num2str(round(Best_pos(1)))]);
disp(['最优最大训练周期为   ',num2str(round(Best_pos(2)))]);
disp(['最优初始学习率为   ',num2str((Best_pos(3)))]);
disp(['最优L2正则化系数为   ',num2str((Best_pos(2)))]);

% SSA优化后的LSTM做预测   对每个分量建模
for d=1:c
data_imf=imf(d,:);
disp(['第',num2str(d),'个分量建模'])    
[x_imf,y_imf]=data_process(data_imf,24);%前24个时刻 预测下一个时刻
%归一化
[xs_imf,mappingx_imf]=mapminmax(x_imf',0,1);x_imf=xs_imf';
[ys_imf,mappingy_imf]=mapminmax(y_imf',0,1);y_imf=ys_imf';
%划分数据
n=size(x_imf,1);
m=round(n*0.7);%前70%训练，对最后30%进行预测

%归一化后的训练和测试集划分
xTrain_imf=x_imf(1:m,:)';
xTest_imf=x_imf(m+1:end,:)';
yTrain_imf=y_imf(1:m,:)';
yTest_imf=y_imf(m+1:end,:)';

%% 创建LSTM回归网络，序列预测，因此，输入24维，输出一维
numFeatures = size(xTrain_imf,1);
numResponses = 1;

%% 基础LSTM测试
numHiddenUnits = round(Best_pos(1));
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
 
%指定训练选项
options = trainingOptions('adam', ...
    'MaxEpochs',round(Best_pos(2)), ...
    'ExecutionEnvironment' ,'cpu',...
    'GradientThreshold',1, ...
    'InitialLearnRate',Best_pos(3), ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',round(0.8*Best_pos(2)), ...
    'LearnRateDropFactor',0.2, ...%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率
    'L2Regularization',0.0001,...
    'Verbose',0);
%训练LSTM
net = trainNetwork(xTrain_imf,yTrain_imf,layers,options);


%训练集测试
numTimeStepsTrain = size(xTrain_imf,2);
for i = 1:numTimeStepsTrain
    [net,predictTrain_imf(:,i)] = predictAndUpdateState(net,xTrain_imf(:,i),'ExecutionEnvironment','cpu');
end

%测试集测试
numTimeStepsTest = size(xTest_imf,2);
for i = 1:numTimeStepsTest
    [net,predictTest_imf(:,i)] = predictAndUpdateState(net,xTest_imf(:,i),'ExecutionEnvironment','cpu');
    %参数更新太多，不能用predict，用predictAndUpdateState
end


% 反归一化
PredictTrain_imf0(d,:)=mapminmax('reverse',predictTrain_imf,mappingy_imf);
PredictTest_imf0(d,:)=mapminmax('reverse',predictTest_imf,mappingy_imf);
YTrain_imf0(d,:)=mapminmax('reverse',yTrain_imf,mappingy_imf);
YTest_imf0(d,:)=mapminmax('reverse',yTest_imf,mappingy_imf);

end
% 各分量预测的结果相加
PredictTrain_imf0=sum(PredictTrain_imf0);
PredictTest_imf0=sum(PredictTest_imf0);
YTrain_imf0=sum(YTrain_imf0);
YTest_imf0=sum(YTest_imf0);


disp('')
disp('训练集误差指标:')
[mae5,rmse5,mape5,error5]=calc_error(YTrain_imf0,PredictTrain_imf0);
fprintf('\n')

disp('测试集误差指标:')
[mae6,rmse6,mape6,error6]=calc_error(YTest_imf0,PredictTest_imf0);
fprintf('\n')

%% 三种模型测试集结果绘图对比

figure
plot(YTest,'k','linewidth',3);
hold on;
plot(PredictTest,'b','linewidth',3);
hold on;
plot(PredictTest_imf,'g','linewidth',3);
hold on;
plot(PredictTest_imf0,'r','linewidth',3);
legend('Target','LSTM','VMD-LSTM','VMD-SSA-LSTM');
title('三种模型测试集结果对比图');
xlabel('Sample Index');
xlabel('Wind Speed');
grid on;

figure
plot(error2,'k','linewidth',3);
hold on
plot(error4,'g','linewidth',3);
hold on
plot(error6,'r','linewidth',3);
legend('LSTM-Error','VMD-LSTM-Eoor','VMD-SSA-LSTM-Eoor');
title(['LSTM-RMSE = ' num2str(rmse2), 'VMD-LSTM-RMSE = ' num2str(rmse4), 'VMD-SSA-LSTM-RMSE = ' num2str(rmse6)]);
grid on;
toc

