function [mae,rmse,mape,error]=calc_error(x1,x2)

 error=x2-x1;  %�������
 rmse=sqrt(mean(error.^2));
 disp(['��������(RMSE)��',num2str(rmse)])

 mae=mean(abs(error));
disp(['ƽ��������MAE����',num2str(mae)])

 mape=mean(abs(error)/x1);
 disp(['ƽ����԰ٷ���MAPE����',num2str(mape*100),'%'])
end

