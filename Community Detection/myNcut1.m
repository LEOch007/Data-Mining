clear all;
close all;
%输入
load('D:\~大三下\数据挖掘实验\lab2\dataset\texas.mat'); %数据 A-邻接矩阵 F-特征向量表 label-真实类标
K = 5; %类的个数
KK = 8; %Kmeans预处理类个数
sigma = 0.4; %高斯核函数的超参数

%输出
C = -1*ones(length(A),1); %类标

%构建相似度矩阵
W = zeros(length(A),length(A));
for i = 1:length(A)
    for j = 1:i-1
%         W(i,j) = 1-pdist2(F(i,:),F(j,:),'cosine'); %余弦距离
        W(i,j) = exp( - pdist2(F(i,:),F(j,:),'cosine') / (2*sigma*sigma) ); %高斯核函数
        W(j,i) = W(i,j);
    end
end

%构建度矩阵
D = eye(length(A));
D = D .* sum(W,2); %根据相似度计算度矩阵

%得到标准化拉普拉斯矩阵
L = D-W;
L = D^(-.5)*L*D^(-.5);

%得到特征矩阵
[eigVector,eigvalue] = eig(L); %特征向量矩阵 & 特征值矩阵
[eigvalue,index] = sort(diag(eigvalue));
FM = eigVector(:,index(2:KK+1));
FM = mynormalize(FM); %按行标准化

%对特征矩阵进行聚类
rng('default'); %保证实验可重复性
CC = kmeans(FM,KK); % 默认距离 -欧式 默认初始方式 -Kmeans++ 默认最大迭代次数 -100

%合并策略
while KK~=K
    Nmin = 100000000; combine = [-1, -1];
    
    %所有合并组合情况
    for i = 1:KK
        for j = 1:i-1
            Ccopy = CC;
            Ccopy(Ccopy==i) = j; %拟合并
            Ccluster = unique(Ccopy); %合并后的所有类标

            Ncut = 0; %合并后的Ncut
            for l = 1:length(Ccluster) %遍历合并后的所有类标
                Ncut = Ncut + ( sum(sum(W(Ccopy==Ccluster(l),Ccopy~=Ccluster(l)))) /  sum(sum(W(Ccopy==Ccluster(l),:))) );
            end
            if Ncut<Nmin %最小化Ncut的合并组合方式
                Nmin = Ncut;
                combine = [i,j];
            end
        end
    end
    CC(CC==combine(1)) = combine(2); %真实合并
    KK = KK-1;
end

%评价聚类效果
ClusteringMeasure(label,CC)