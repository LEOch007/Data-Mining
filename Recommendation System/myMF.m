%数据集1
temp = load('D:\~大三下\数据挖掘实验\lab4\Lab4_数据集\movielens数据集\ml-100k\ml-100k\u.data');
data = temp(:,1:3); 
clear temp;
%划分数据集
data_train = data(1:round(length(data)*0.8),:); %训练集
data_test = data(round(length(data)*0.8)+1:end,:); %测试集

%参数
d = 30; %维度
maxIter = 20; %迭代次数
alpha = 0.01;
lambda = 10;

%初始化UV
num_u = max(data(:,1));
num_i = max(data(:,2));
rng(10); U = rand(num_u,d);
rng(10); V = rand(num_i,d);
U_last = U;
V_last = V;

%随机梯度下降法更新UV
for limit = 1:maxIter
    %遍历数据集
    for i = 1:size(data_train,1)
        tempUser = data_train(i,1);  %用户
        tempItem = data_train(i,2);  %商品
        tempGrade = data_train(i,3); %真实分数
        E = tempGrade - U(tempUser,:) * V(tempItem,:)'; %差值
        U(tempUser,:) = U_last(tempUser,:) + alpha * (E*V_last(tempItem,:) - lambda*U_last(tempUser,:)); %更新
        V(tempItem,:) = V_last(tempItem,:) + alpha * (E*U_last(tempUser,:) - lambda*V_last(tempItem,:)); %更新
    end
    U_last = U;
    V_last = V;
    alpha = alpha/(1+limit); %动态学习率
end

%预测结果
R_prect = U*V';
actual = data_test(:,3);
prect = -1*ones(size(data_test,1),1);
for l = 1:size(data_test,1)
    prect(l) = R_prect(data_test(l,1),data_test(l,2));
end

%评估
mae = MAE(prect,actual)
rmse = RMSE(prect,actual)