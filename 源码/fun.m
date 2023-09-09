%适应度函数
%mse作为适应度值
function [fitness,net] = fun(x,numFeatures,numResponses,xTrain,yTrain,xTest,yTest) 

numHiddenUnits = round(x(1));%LSTM网路包含的隐藏单元数目
maxEpochs = round(x(2));%最大训练周期
InitialLearnRate = x(3);%初始学习率


%设置网络
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%指定训练选项，采用cpu训练， 这里用cpu是为了保证能直接运行，如果需要gpu训练，改成gpu就行了，且保证cuda有安装
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'ExecutionEnvironment' ,'cpu',...
    'InitialLearnRate',InitialLearnRate,...
    'GradientThreshold',1, ...
    'LearnRateDropPeriod',75, ...
    'LearnRateDropFactor',0.2, ...%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率
    'L2Regularization',0.0001, ...
    'Verbose',0);
%'Plots','training-progress'
%训练LSTM
net = trainNetwork(xTrain,yTrain,layers,options);


%测试集测试
numTimeStepsTest = size(xTest,2);
for i = 1:numTimeStepsTest
    [net,predictTest_fit(:,i)] = predictAndUpdateState(net,xTest(:,i),'ExecutionEnvironment','cpu');
    %参数更新太多，不能用predict，用predictAndUpdateState
end

% % 反归一化
% 
% PredictTest_fit=mapminmax('reverse',predictTest_fit,mappingy);
% YTest_fit=mapminmax('reverse',yTest,mappingy);
% 
% %测试集rmse
% fitness = sqrt(mse(YTest_fit-PredictTest_fit));

%如果不归一化比较
fitness = mse(yTest-predictTest_fit);
disp('一次训练结束....')
end