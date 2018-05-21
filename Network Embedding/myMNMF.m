clear;
close all;
%输入
load('D:\~大三下\数据挖掘实验\lab3\数据集\texas'); %数据 A-邻接矩阵 F-特征向量表 label-真实类标
k = 5; %社区个数
m = 5; %降维后的维数
K = 8; %KNN参数
lrate = 0.02; %学习率
imax = 300; %迭代次数
alpha = 0.01; belta = 0.012; lamda = 1; %超参数

%图嵌入学习 
%M-NMF

%得到S矩阵
S1 = A; %一阶相似度
S2 = zeros(length(A),length(A)); %二阶相似度
for i = 1:length(A)
    for j = 1:i-1
        S2(i,j) = 1-pdist2(A(i,:),A(j,:),'cosine');
        S2(j,i) = S2(i,j);
    end
end
S = S1 + 5*S2; %S矩阵
clear S1 S2;

%得到B矩阵
B = zeros(length(A),length(A)); %B矩阵
e = sum(sum(A))/2; %总权重
for i = 1:length(A)
    for j = 1:length(A)
        ki = sum(A(i,:)); %i节点的度
        kj = sum(A(j,:));
        B(i,j) = A(i,j) - (ki*kj)/(2*e);
    end
end
clear ki kj e;

%初始化
n = length(A);
rng(2); M = rand(n,m);
rng(3); U = rand(n,m); %n*m非负矩阵
rng(5); C = rand(k,m);
rng(6); H = rand(n,k);

%迭代更新 梯度下降
iteration = 0;
while true
    M_temp = lrate* M.* (S*U./ (M*U'*U));
    U_temp = lrate* U.* ((S'*M + alpha*H*C)./ (U*(M'*M + alpha*C'*C)));
    C_temp = lrate* C.* (H'*U./ (C*U'*U));
    
    Delta = (2*belta*(B*H)).* (2*belta*(B*H)) + 16*lamda*(H*H'*H).* (2*belta*A*H + 2*alpha*U*C' + (4*lamda-2*alpha)*H);
    H_temp = H.* sqrt( (-2*belta*B*H + sqrt(Delta))./ (8*lamda*H*H'*H) );
    
    %同步更新
    M = M_temp;
    U = U_temp;
    C = C_temp;
    H = H_temp;
    
    %终止条件
    iteration = iteration+1;
    if iteration >= imax
        break;
    end
end
clear M_temp U_temp H_temp C_temp Delta;

%划分数据集
rng(10); randidx = randperm(length(U));  %打乱原数据集顺序
pos = round(length(randidx)*0.7); %七三分
trainidx = randidx(1:pos); %训练集下标
prectidx = randidx(pos+1:end); %测试集下标

X = U(trainidx,:); Y = label(trainidx); %训练集特征/标签
prectX = U(prectidx,:); %测试集特征
valiY = label(prectidx); prectY = zeros(length(prectX),1); %真实标签/预测标签

%KNN
for i = 1:length(prectX)
    prectY(i) = myKNN(prectX(i,:),X,Y,K);
end

%分类效果
classificationACC(valiY,prectY)