%数据集1
temp = load('D:\~大三下\数据挖掘实验\lab4\Lab4_数据集\movielens数据集\ml-100k\ml-100k\u.data');
data = temp(:,1:3); 
clear temp;
%划分数据集
data_train = data(1:round(length(data)*0.8),:); %训练集
data_test = data(round(length(data)*0.8)+1:end,:); %测试集

%构建 用户-商品评分矩阵
num_u = max(data(:,1));
num_i = max(data(:,2));
R = -1*ones(num_u,num_i);
for i = 1:size(data_train,1)
    R(data_train(i,1),data_train(i,2)) = data_train(i,3);
end

%构建 评分偏差矩阵
dev = zeros(num_i,num_i);
for i = 1:num_i
    for j = 1:i-1
        same = intersect(find(R(:,i)~=-1),find(R(:,j)~=-1)); %同时评分的用户
        if isempty(same)
            %没有同时对这两个商品评分的用户
            dev(i,j) = -1000;
        else
            %计算评分偏差
            dev(i,j) = sum(R(same,i)-R(same,j))/length(same);
        end
        dev(j,i) = dev(i,j); %---
    end
end
            
%预测用户未评分商品
mae = -1*ones(num_u,1);
rmse = -1*ones(num_u,1);
for i = 1:num_u
    goods = data_test(data_test(:,1)==i,2); %用户i待评分商品
    if isempty(goods)
        continue;
    end
    rgood = find(R(i,:)~=-1); %用户i有评分的商品
    score = -1*ones(length(goods),1); %预测的分数
    for j = 1:length(goods)
        sgood = intersect(rgood,find(dev(goods(j),:)~=-1000)); %与商品goods(j)有评分偏差的商品
        if isempty(sgood) %空 
            score(j) = mean(R(i,rgood)); %取用户评分均值
        else 
            score(j) = sum(dev(goods(j),sgood)+R(i,sgood))/length(sgood); %偏差取值
        end
    end
    
    %测试集
    actual = data_test(data_test(:,1)==i,3); %实际评分
    mae(i) = MAE(score,actual); %MAE误差
    rmse(i) = RMSE(score,actual); %RMSE误差
end

%评估结果
m_mae = mean(mae);
m_rmse = mean(rmse);
