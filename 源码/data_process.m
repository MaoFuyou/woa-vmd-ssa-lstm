function [in,out]=data_process(data,num)
% ����1-num��Ϊ���� ��num+1��Ϊ���
n=length(data)-num;
for i=1:n
    x(i,:)=data(i:i+num);
end
in=x(:,1:end-1);
out=x(:,end);