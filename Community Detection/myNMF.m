clear;
close all;
%输入
load('D:\~大三下\数据挖掘实验\lab2\dataset\washington.mat'); %数据 A-邻接矩阵 F-特征向量表 label-真实类标
K = 5; %类的个数
alpha = 0.06; %步长
%输出
C = 1:length(A); %类标 初始化为各自不同社区

%随机初始化UV矩阵
N = length(A); %数据个数
rng(10); %保证实验可重复性
U = rand(N,K); %N*K非负矩阵
rng(3);
V = rand(N,K);

%迭代更新UV矩阵
iteration = 0;
while true
    U_temp = U; V_temp = V; %防止同步更新
    AV = A*V; UVV = U*V'*V;
    AU = A'*U; VUU = V*U'*U;
    %U
    for i = 1:N
        for j = 1:K
            U_temp(i,j) = U(i,j)*AV(i,j)/UVV(i,j); %异步更新
        end
    end
    %V
    for i = 1:N
        for j = 1:K
            V_temp(i,j) = V(i,j)*AU(i,j)/VUU(i,j); %异步更新
        end
    end
    %更新UV
    U = U_temp; V = V_temp; 
    iteration = iteration+1;
    
    %终止条件
    if iteration >= 100
        break;
    end
end

%分配类标
% [mvalue,midx] = max(U,[],2); %U每一行最大值
% C = midx; %分配类标
C = kmeans(U,K);

ClusteringMeasure(label,C) %评估结果