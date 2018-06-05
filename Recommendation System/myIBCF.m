% %小数据集
% data = load ('D:\~大三下\数据挖掘实验\lab4\Lab4_数据集\很小的数据集\train_small.txt');
% %小数据集2
% data = load ('D:\~大三下\数据挖掘实验\lab4\Lab4_数据集\很小的数据集\train_small_2.txt'); 
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

%构建 商品相似度矩阵
S = ones(num_i,num_i);
for i = 1:num_i
    for j = 1:i-1
        same = intersect(find(R(:,i)~=-1),find(R(:,j)~=-1)); %同时评分的用户
        if isempty(same)
            %没有同时对这两个商品评分的用户
            S(i,j) = 0;
        else
            %皮尔逊相关系数
            ri = mean(R(same,i)); rj = mean(R(same,j));
            fz = sum( (R(same,i)-ri).*(R(same,j)-rj) );
            if fz==0
                S(i,j) = 0; %防止NaN的出现
            else
                fm = sqrt(sum( (R(same,i)-ri).^2 )) * sqrt(sum( (R(same,j)-rj).^2 ));
                S(i,j) = fz / fm;
            end
        end
        S(j,i) = S(i,j);
    end
end

%预测用户未评分商品
N = 20;
alpha = 0.005; %平滑因子
mae = -1*ones(num_u,1);
rmse = -1*ones(num_u,1);
for i = 1:num_u
    goods = data_test(data_test(:,1)==i,2); %未评分商品
    if isempty(goods)
        continue;
    end
    rgood = find(R(i,:)~=-1); %评分商品
    score = -1*ones(length(goods),1); %预测的分数
    for j = 1:length(goods)
        %找近邻
        Neighbors = zeros(N,1); %近邻
        count = 1; %近邻数量
        [~,id] = sort(S(goods(j),:),'descend'); %降序排列
        for l=1:length(id)
            if count > N %大于最大近邻数
                break;
            end
            if ismember(id(l),rgood) %如果已评分
                Neighbors(count) = id(l);
                count = count+1;
            end
        end

        Rating = R(i,Neighbors(1:count-1)); %相似商品评分
        Similar = S(goods(j),Neighbors(1:count-1)); %相似度
        score(j) = (sum(Similar.*Rating)+alpha)/(sum(Similar)+(count-1)*alpha); %预测的分数
    end
    
    %测试集
    test = data_test(data_test(:,1)==i,:);
    actual = test(:,3); %实际评分
    prect = score; %预测评分
    
    mae(i) = MAE(prect,actual); %MAE误差
    rmse(i) = RMSE(prect,actual); %RMSE误差
end

%评估结果
m_mae = mean(mae);
m_rmse = mean(rmse);
        