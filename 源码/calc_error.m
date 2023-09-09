function [mae,rmse,mape,error]=calc_error(x1,x2)

 error=x2-x1;  %计算误差
 rmse=sqrt(mean(error.^2));
 disp(['根均方差(RMSE)：',num2str(rmse)])

 mae=mean(abs(error));
disp(['平均绝对误差（MAE）：',num2str(mae)])

 mape=mean(abs(error)/x1);
 disp(['平均相对百分误差（MAPE）：',num2str(mape*100),'%'])
end

