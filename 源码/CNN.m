%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  �������ݣ�ʱ�����еĵ������ݣ�
%result = xlsread('���ݼ�.xlsx');
%result = xlsread('Data2.xlsx','E3:E357');
result = xlsread('Data3.xlsx','B1:B355');
%%  ���ݷ���
num_samples = length(result);  % �������� 
kim = 15;                      % ��ʱ������kim����ʷ������Ϊ�Ա�����
zim =  1;                      % ��zim��ʱ������Ԥ��

%%  �������ݼ�
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(result(i: i + kim - 1), 1, kim), result(i + kim + zim - 1)];
end

%%  ����ѵ�����Ͳ��Լ�
temp = 1: 1: 338;

P_train = res(temp(1: 300), 1: 15)';
T_train = res(temp(1: 300), 16)';
M = size(P_train, 2);

P_test = res(temp(301: end), 1: 15)';
T_test = res(temp(301: end), 16)';
N = size(P_test, 2);

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  ����ƽ��
%   ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
%   Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
%   ����Ӧ��ʼ�պ���������ݽṹ����һ��
p_train =  double(reshape(p_train, 15, 1, 1, M));
p_test  =  double(reshape(p_test , 15, 1, 1, N));
t_train =  double(t_train)';
t_test  =  double(t_test )';

%%  ��������ṹ
layers = [
 imageInputLayer([15, 1, 1])     % ����� �������ݹ�ģ[15, 1, 1]
 
 convolution2dLayer([3, 1], 16)  % ����˴�С 3*1 ����16������ͼ
 batchNormalizationLayer         % ����һ����
 reluLayer                       % Relu�����
 
 convolution2dLayer([3, 1], 32)  % ����˴�С 3*1 ����32������ͼ
 batchNormalizationLayer         % ����һ����
 reluLayer                       % Relu�����
 

 fullyConnectedLayer(1)          % ȫ���Ӳ�
 regressionLayer];               % �ع��

%%  ��������
options = trainingOptions('adam', ...      % Adam �ݶ��½��㷨
    'MaxEpochs', 60000, ...                  % ���ѵ������ 300
    'InitialLearnRate', 1e-2, ...          % ��ʼѧϰ��Ϊ0.01
    'LearnRateSchedule', 'piecewise', ...  % ѧϰ���½�
    'LearnRateDropFactor', 0.1, ...        % ѧϰ���½����� 0.1
    'LearnRateDropPeriod', 400, ...        % ����200��ѵ���� ѧϰ��Ϊ 0.01 * 0.1
    'Shuffle', 'every-epoch', ...          % ÿ��ѵ���������ݼ�
    'Plots', 'training-progress', ...      % ��������
    'Verbose', 1);

%%  ѵ��ģ��
net = trainNetwork(p_train, t_train, layers, options);

%%  ģ��Ԥ��
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  ���������
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

%%  �����������ͼ
analyzeNetwork(layers)

%%  ��ͼ
figure
plot(1: M, T_train, 'r-', 1: M, T_sim1, 'b-', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-', 1: N, T_sim2, 'b-', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  ���ָ�����
%  R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2')^2 / norm(T_test -  mean(T_test ))^2;

disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;

disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1)])
disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2)])