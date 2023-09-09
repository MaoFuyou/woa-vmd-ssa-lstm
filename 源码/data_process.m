function [in,out]=data_process(data,num)
% 采用1-num作为输入 第num+1作为输出
n=length(data)-num;
for i=1:n
    x(i,:)=data(i:i+num);
end
in=x(:,1:end-1);
out=x(:,end);