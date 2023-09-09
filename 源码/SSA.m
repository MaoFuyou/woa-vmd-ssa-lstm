%_________________________________________________________________________%
% 麻雀优化算法             %
%_________________________________________________________________________%
function [Best_pos,Best_score,curve,BestNet]=SSA(pop,Max_iter,lb,ub,dim,fobj)
disp('麻雀算法开始...')
ST = 0.6;%预警值
PD = 0.7;%发现者的比列，剩下的是加入者
SD = 0.2;%意识到有危险麻雀的比重

PDNumber = round(pop*PD); %发现者数量
SDNumber = round(pop*SD);%意识到有危险麻雀数量
if(max(size(ub)) == 1)
   ub = ub.*ones(1,dim);
   lb = lb.*ones(1,dim);  
end
net = {};
%种群初始化
X0=initialization(pop,dim,ub,lb);
X = X0;
%计算初始适应度值
fitness = zeros(1,pop);
for i = 1:pop
   [fitness(i),net{i}] =  fobj(X(i,:));
end
 [fitness, index]= sort(fitness);%排序
BestF = fitness(1);
WorstF = fitness(end);
GBestF = fitness(1);%全局最优适应度值
for i = 1:pop
    X(i,:) = X0(index(i),:);
    net{i}=net{index(i)};
end
curve=zeros(1,Max_iter);
GBestX = X(1,:);%全局最优位置
X_new = X;
BestNet = net{1};
curve(1)=GBestF;
for i = 2: Max_iter
    disp(['第',num2str(i),'次迭代']);
    BestF = fitness(1);
    WorstF = fitness(end);

    
    R2 = rand(1);
   for j = 1:PDNumber
      if(R2<ST)
          X_new(j,:) = X(j,:).*exp(-j/(rand(1)*Max_iter));
      else
          X_new(j,:) = X(j,:) + randn()*ones(1,dim);
      end     
   end
   for j = PDNumber+1:pop
%        if(j>(pop/2))
        if(j>(pop - PDNumber)/2 + PDNumber)
          X_new(j,:)= randn().*exp((X(end,:) - X(j,:))/j^2);
       else
          %产生-1，1的随机数
          A = ones(1,dim);
          for a = 1:dim
            if(rand()>0.5)
                A(a) = -1;
            end
          end 
          AA = A'*inv(A*A');     
          X_new(j,:)= X(1,:) + abs(X(j,:) - X(1,:)).*AA';
       end
   end
   Temp = randperm(pop);
   SDchooseIndex = Temp(1:SDNumber); 
   for j = 1:SDNumber
       if(fitness(SDchooseIndex(j))>BestF)
           X_new(SDchooseIndex(j),:) = X(1,:) + randn().*abs(X(SDchooseIndex(j),:) - X(1,:));
       elseif(fitness(SDchooseIndex(j))== BestF)
           K = 2*rand() -1;
           X_new(SDchooseIndex(j),:) = X(SDchooseIndex(j),:) + K.*(abs( X(SDchooseIndex(j),:) - X(end,:))./(fitness(SDchooseIndex(j)) - fitness(end) + 10^-8));
       end
   end
   %边界控制
   for j = 1:pop
       for a = 1: dim
           if(X_new(j,a)>ub(a)||isnan(X_new(j,a)))%sNaN()函数用来确定一个值是否为NaN,如果是NaN就取上边界值
               X_new(j,a) =ub(a);
           end
           if(X_new(j,a)<lb(a)||isnan(X_new(j,a)))
               X_new(j,a) =lb(a);
           end
       end
   end 
   %更新位置
   for j=1:pop
    [fitness_new(j),net{j}] = fobj(X_new(j,:));
   end
   for j = 1:pop
    if(fitness_new(j) < GBestF)
       GBestF = fitness_new(j);
        GBestX = X_new(j,:); 
        BestNet=net{j};
    end
   end
   X = X_new;
   fitness = fitness_new;
    %排序更新
   [fitness, index]= sort(fitness);%排序
   BestF = fitness(1);
   WorstF = fitness(end);
   for j = 1:pop
      X(j,:) = X(index(j),:);
      net{j}=net{index(j)};
   end
   curve(i) = GBestF;
end
Best_pos =GBestX;
Best_score = curve(end);
end



