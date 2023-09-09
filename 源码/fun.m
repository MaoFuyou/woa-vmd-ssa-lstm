%��Ӧ�Ⱥ���
%mse��Ϊ��Ӧ��ֵ
function [fitness,net] = fun(x,numFeatures,numResponses,xTrain,yTrain,xTest,yTest) 

numHiddenUnits = round(x(1));%LSTM��·���������ص�Ԫ��Ŀ
maxEpochs = round(x(2));%���ѵ������
InitialLearnRate = x(3);%��ʼѧϰ��


%��������
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%ָ��ѵ��ѡ�����cpuѵ���� ������cpu��Ϊ�˱�֤��ֱ�����У������Ҫgpuѵ�����ĳ�gpu�����ˣ��ұ�֤cuda�а�װ
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'ExecutionEnvironment' ,'cpu',...
    'InitialLearnRate',InitialLearnRate,...
    'GradientThreshold',1, ...
    'LearnRateDropPeriod',75, ...
    'LearnRateDropFactor',0.2, ...%ָ����ʼѧϰ�� 0.005���� 125 ��ѵ����ͨ���������� 0.2 ������ѧϰ��
    'L2Regularization',0.0001, ...
    'Verbose',0);
%'Plots','training-progress'
%ѵ��LSTM
net = trainNetwork(xTrain,yTrain,layers,options);


%���Լ�����
numTimeStepsTest = size(xTest,2);
for i = 1:numTimeStepsTest
    [net,predictTest_fit(:,i)] = predictAndUpdateState(net,xTest(:,i),'ExecutionEnvironment','cpu');
    %��������̫�࣬������predict����predictAndUpdateState
end

% % ����һ��
% 
% PredictTest_fit=mapminmax('reverse',predictTest_fit,mappingy);
% YTest_fit=mapminmax('reverse',yTest,mappingy);
% 
% %���Լ�rmse
% fitness = sqrt(mse(YTest_fit-PredictTest_fit));

%�������һ���Ƚ�
fitness = mse(yTest-predictTest_fit);
disp('һ��ѵ������....')
end