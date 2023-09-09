clc;
clear 
close all

%% ��һ��LSTMԤ��
tic
disp('��������������������������������������������������������������������������������������������')
disp('��һ��LSTMԤ��')
disp('��������������������������������������������������������������������������������������������')
data = xlsread('Data1.xlsx','D3:D357');
[x,y]=data_process(data,24);%ǰ24��ʱ�� Ԥ����һ��ʱ��
%��һ��
[xs,mappingx]=mapminmax(x',0,1);x=xs';
[ys,mappingy]=mapminmax(y',0,1);y=ys';
%��������
n=size(x,1);
m=round(n*0.7);%ǰ70%ѵ���������30%����Ԥ��

%��һ�����ѵ���Ͳ��Լ�����
xTrain=x(1:m,:)';
xTest=x(m+1:end,:)';
yTrain=y(1:m,:)';
yTest=y(m+1:end,:)';

%% ����LSTM�ع����磬����Ԥ�⣬��ˣ�����24ά�����һά
numFeatures = size(xTrain,1);
numResponses = 1;

%% ����LSTM����
numHiddenUnits = 35;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
 
%ָ��ѵ��ѡ��
options = trainingOptions('adam', ...
    'MaxEpochs',35, ...
    'ExecutionEnvironment' ,'cpu',...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'LearnRateDropFactor',0.2, ...%ָ����ʼѧϰ�� 0.005���� 125 ��ѵ����ͨ���������� 0.2 ������ѧϰ��
    'L2Regularization',0.0001,...
    'Verbose',0);
%ѵ��LSTM
net = trainNetwork(xTrain,yTrain,layers,options);


%ѵ��������
numTimeStepsTrain = size(xTrain,2);
for i = 1:numTimeStepsTrain
    [net,predictTrain(:,i)] = predictAndUpdateState(net,xTrain(:,i),'ExecutionEnvironment','cpu');
end

%���Լ�����
numTimeStepsTest = size(xTest,2);
for i = 1:numTimeStepsTest
    [net,predictTest(:,i)] = predictAndUpdateState(net,xTest(:,i),'ExecutionEnvironment','cpu');
    %��������̫�࣬������predict����predictAndUpdateState
end


% ����һ��
PredictTrain=mapminmax('reverse',predictTrain,mappingy);
PredictTest=mapminmax('reverse',predictTest,mappingy);
YTrain=mapminmax('reverse',yTrain,mappingy);
YTest=mapminmax('reverse',yTest,mappingy);

disp('')
disp('ѵ�������ָ��:')
[mae1,rmse1,mape1,error1]=calc_error(YTrain,PredictTrain);
fprintf('\n')

disp('���Լ����ָ��:')
[mae2,rmse2,mape2,error2]=calc_error(YTest,PredictTest);
fprintf('\n')
toc


%% VMD-LSTMԤ��

tic
disp('��������������������������������������������������������������������������������������������')
disp('VMD-LSTMԤ��')
disp('��������������������������������������������������������������������������������������������')
load vmd_data
imf=u;
c=size(imf,1);
%% ��ÿ��������ģ
for d=1:c
disp(['��',num2str(d),'��������ģ'])    
data_imf=imf(d,:);
[x_imf,y_imf]=data_process(data_imf,24);%ǰ24��ʱ�� Ԥ����һ��ʱ��
%��һ��
[xs_imf,mappingx_imf]=mapminmax(x_imf',0,1);x_imf=xs_imf';
[ys_imf,mappingy_imf]=mapminmax(y_imf',0,1);y_imf=ys_imf';
%��������
n=size(x_imf,1);
m=round(n*0.7);%ǰ70%ѵ���������30%����Ԥ��

%��һ�����ѵ���Ͳ��Լ�����
xTrain_imf=x_imf(1:m,:)';
xTest_imf=x_imf(m+1:end,:)';
yTrain_imf=y_imf(1:m,:)';
yTest_imf=y_imf(m+1:end,:)';

%% ����LSTM�ع����磬����Ԥ�⣬��ˣ�����24ά�����һά
numFeatures = size(xTrain_imf,1);
numResponses = 1;

%% ����LSTM����
numHiddenUnits = 35;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
 
%ָ��ѵ��ѡ��
options = trainingOptions('adam', ...
    'MaxEpochs',35, ...
    'ExecutionEnvironment' ,'cpu',...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'LearnRateDropFactor',0.2, ...%ָ����ʼѧϰ�� 0.005���� 125 ��ѵ����ͨ���������� 0.2 ������ѧϰ��
    'L2Regularization',0.0001,...
    'Verbose',0);
%ѵ��LSTM
net = trainNetwork(xTrain_imf,yTrain_imf,layers,options);


%ѵ��������
numTimeStepsTrain = size(xTrain_imf,2);
for i = 1:numTimeStepsTrain
    [net,predictTrain_imf(:,i)] = predictAndUpdateState(net,xTrain_imf(:,i),'ExecutionEnvironment','cpu');
end

%���Լ�����
numTimeStepsTest = size(xTest_imf,2);
for i = 1:numTimeStepsTest
    [net,predictTest_imf(:,i)] = predictAndUpdateState(net,xTest_imf(:,i),'ExecutionEnvironment','cpu');
    %��������̫�࣬������predict����predictAndUpdateState
end


% ����һ��
PredictTrain_imf(d,:)=mapminmax('reverse',predictTrain_imf,mappingy_imf);
PredictTest_imf(d,:)=mapminmax('reverse',predictTest_imf,mappingy_imf);
YTrain_imf(d,:)=mapminmax('reverse',yTrain_imf,mappingy_imf);
YTest_imf(d,:)=mapminmax('reverse',yTest_imf,mappingy_imf);

end
% ������Ԥ��Ľ�����
PredictTrain_imf=sum(PredictTrain_imf);
PredictTest_imf=sum(PredictTest_imf);
YTrain_imf=sum(YTrain_imf);
YTest_imf=sum(YTest_imf);


disp('')
disp('ѵ�������ָ��:')
[mae3,rmse3,mape3,error3]=calc_error(YTrain_imf,PredictTrain_imf);
fprintf('\n')

disp('���Լ����ָ��:')
[mae4,rmse4,mape4,error4]=calc_error(YTest_imf,PredictTest_imf);
fprintf('\n')
toc;



%% VMD-SSA-LSTMԤ��
tic
disp('��������������������������������������������������������������������������������������������')
disp('VMD-SSA-LSTMԤ��')
disp('��������������������������������������������������������������������������������������������')
pop=10; % ��ȸ����
Max_iteration=20; % ����������
dim=3; % �Ż�lstm��3������
lb = [40,40,0.001];%�±߽�
ub = [200,200,0.03];%�ϱ߽�
fobj = @(x) fun(x,numFeatures,numResponses,xTrain,yTrain,xTest,yTest);
%������ȸ�㷨
[Best_pos,Best_score,SSA_curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj); %��ʼ�Ż�

%% ���ƽ�������
figure
plot(SSA_curve,'r-','linewidth',3)
xlabel('��������')
ylabel('�������MSE')
legend('�����Ӧ��')
title('SSA-LSTM�Ľ�����������')

disp('')
disp(['�������ص�Ԫ��ĿΪ   ',num2str(round(Best_pos(1)))]);
disp(['�������ѵ������Ϊ   ',num2str(round(Best_pos(2)))]);
disp(['���ų�ʼѧϰ��Ϊ   ',num2str((Best_pos(3)))]);
disp(['����L2����ϵ��Ϊ   ',num2str((Best_pos(2)))]);

% SSA�Ż����LSTM��Ԥ��   ��ÿ��������ģ
for d=1:c
data_imf=imf(d,:);
disp(['��',num2str(d),'��������ģ'])    
[x_imf,y_imf]=data_process(data_imf,24);%ǰ24��ʱ�� Ԥ����һ��ʱ��
%��һ��
[xs_imf,mappingx_imf]=mapminmax(x_imf',0,1);x_imf=xs_imf';
[ys_imf,mappingy_imf]=mapminmax(y_imf',0,1);y_imf=ys_imf';
%��������
n=size(x_imf,1);
m=round(n*0.7);%ǰ70%ѵ���������30%����Ԥ��

%��һ�����ѵ���Ͳ��Լ�����
xTrain_imf=x_imf(1:m,:)';
xTest_imf=x_imf(m+1:end,:)';
yTrain_imf=y_imf(1:m,:)';
yTest_imf=y_imf(m+1:end,:)';

%% ����LSTM�ع����磬����Ԥ�⣬��ˣ�����24ά�����һά
numFeatures = size(xTrain_imf,1);
numResponses = 1;

%% ����LSTM����
numHiddenUnits = round(Best_pos(1));
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
 
%ָ��ѵ��ѡ��
options = trainingOptions('adam', ...
    'MaxEpochs',round(Best_pos(2)), ...
    'ExecutionEnvironment' ,'cpu',...
    'GradientThreshold',1, ...
    'InitialLearnRate',Best_pos(3), ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',round(0.8*Best_pos(2)), ...
    'LearnRateDropFactor',0.2, ...%ָ����ʼѧϰ�� 0.005���� 125 ��ѵ����ͨ���������� 0.2 ������ѧϰ��
    'L2Regularization',0.0001,...
    'Verbose',0);
%ѵ��LSTM
net = trainNetwork(xTrain_imf,yTrain_imf,layers,options);


%ѵ��������
numTimeStepsTrain = size(xTrain_imf,2);
for i = 1:numTimeStepsTrain
    [net,predictTrain_imf(:,i)] = predictAndUpdateState(net,xTrain_imf(:,i),'ExecutionEnvironment','cpu');
end

%���Լ�����
numTimeStepsTest = size(xTest_imf,2);
for i = 1:numTimeStepsTest
    [net,predictTest_imf(:,i)] = predictAndUpdateState(net,xTest_imf(:,i),'ExecutionEnvironment','cpu');
    %��������̫�࣬������predict����predictAndUpdateState
end


% ����һ��
PredictTrain_imf0(d,:)=mapminmax('reverse',predictTrain_imf,mappingy_imf);
PredictTest_imf0(d,:)=mapminmax('reverse',predictTest_imf,mappingy_imf);
YTrain_imf0(d,:)=mapminmax('reverse',yTrain_imf,mappingy_imf);
YTest_imf0(d,:)=mapminmax('reverse',yTest_imf,mappingy_imf);

end
% ������Ԥ��Ľ�����
PredictTrain_imf0=sum(PredictTrain_imf0);
PredictTest_imf0=sum(PredictTest_imf0);
YTrain_imf0=sum(YTrain_imf0);
YTest_imf0=sum(YTest_imf0);


disp('')
disp('ѵ�������ָ��:')
[mae5,rmse5,mape5,error5]=calc_error(YTrain_imf0,PredictTrain_imf0);
fprintf('\n')

disp('���Լ����ָ��:')
[mae6,rmse6,mape6,error6]=calc_error(YTest_imf0,PredictTest_imf0);
fprintf('\n')

%% ����ģ�Ͳ��Լ������ͼ�Ա�

figure
plot(YTest,'k','linewidth',3);
hold on;
plot(PredictTest,'b','linewidth',3);
hold on;
plot(PredictTest_imf,'g','linewidth',3);
hold on;
plot(PredictTest_imf0,'r','linewidth',3);
legend('Target','LSTM','VMD-LSTM','VMD-SSA-LSTM');
title('����ģ�Ͳ��Լ�����Ա�ͼ');
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

