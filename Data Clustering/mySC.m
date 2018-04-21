clear all;
close all;
%输入
X = load('D:\~大三下\数据挖掘实验\lab1\datasets\Aggregation_cluster=7.txt'); %数据 
sigma = 0.29; %高斯核函数的超参数
K = 7; %类的个数
% rng('shuffle');
% sigma = 0.1 + (10-0.1).*rand(1); %随机选取参数

%输出
C = -1*ones(length(X),1); %类标

%构建相似度矩阵
W = zeros(length(X),length(X)); %忽略自身的相似度
for i = 1:length(X)
    for j = 1:i-1
        W(i,j) = exp( - sum((X(i,:)-X(j,:)).*(X(i,:)-X(j,:))) / (2*sigma*sigma) ); %高斯核函数
        W(j,i) = W(i,j); %对称矩阵
    end
end

%构建度矩阵
D = eye(length(X));
D = D .* sum(W,2); %根据相似度计算度矩阵

%得到标准化拉普拉斯矩阵
L = D-W;
L = D^(-.5)*L*D^(-.5);

%得到特征矩阵
[eigVector,eigvalue] = eig(L); %特征向量矩阵 & 特征值矩阵
[eigvalue,index] = sort(diag(eigvalue));
F = eigVector(:,index(2:K+1));
F = mynormalize(F);

%对特征矩阵进行聚类
rng('default'); %保证实验可重复性
C = kmeans(F,K); % 默认距离 -欧式 默认初始方式 -Kmeans++ 默认最大迭代次数 -100

%输出可视化聚类效果
figure;
hold on;
for k=1:K
    px = X(C==k,1);
    py = X(C==k,2);
    plot(px,py,'.');
end