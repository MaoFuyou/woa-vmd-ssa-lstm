tic
clc
clear all
fs=1;%����Ƶ�ʣ���ʱ��������������֮���ʱ������������1h����
Ts=1/fs;%��������
L=1000;%��������,���ж��ٸ�����
t=(0:L-1)*Ts;%ʱ������
STA=0; %������ʼλ�ã������0h��ʼ����

X = xlsread('������.xlsx','o2:o1001');
%X = xlsread('Data2.xlsx','D3:D357');

%--------- some sample parameters forVMD������VMD��Ʒ������������---------------
alpha = 3950;       % moderate bandwidth constraint���ʶȵĴ���Լ��/�ͷ�����
tau = 0;          % noise-tolerance (no strict fidelity enforcement)���������ޣ�û���ϸ�ı����ִ�У�
K = 9;              % modes���ֽ��ģ̬��
DC = 0;             % no DC part imposed����ֱ������
init = 1;           % initialize omegas uniformly  ��omegas�ľ��ȳ�ʼ��
tol = 1e-7         
%--------------- Run actual VMD code:���ݽ���vmd�ֽ�---------------------------
[u, u_hat, omega] = VMD(X, alpha, tau, K, DC, init, tol);

save vmd_data u

figure(1);
imfn=u;
n=size(imfn,1); %size(X,1),���ؾ���X��������size(X,2),���ؾ���X��������N=size(X,2)�����ǰѾ���X��������ֵ��N
subplot(n+1,1,1);  % m�����У�n�����У�p��������ͼ�λ��ڵڼ��С��ڼ��С�����subplot(2,2,[1,2])
plot(t,X); %�����ź�
ylabel('ԭʼ��ͨ����','fontsize',14,'fontname','����');
A = ['r','g','b','c','m','y','k'];
for n1=1:n
    subplot(n+1,1,n1+1);
    plot(t,u(n1,:),A(mod(n1,7)+1),'linewidth',2.5);%���IMF������a(:,n)���ʾ����a�ĵ�n��Ԫ�أ�u(n1,:)��ʾ����u��n1��Ԫ��
    ylabel(['IMF' int2str(n1)]);%int2str(i)�ǽ���ֵi���������ת����ַ���y������
end
 xlabel('ʱ��\itt/hour','fontsize',14,'fontname','����');
 toc;
 %----------------------��������Ƶ��ȷ���ֽ����K-----------------------------
average=mean(omega)%������е�ƽ��ֵ
omega;
