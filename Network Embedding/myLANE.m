clear;
close all;
%输入
load('D:\~大三下\数据挖掘实验\lab3\数据集\texas'); %数据 Network-邻接矩阵 Attributes-特征向量表 Label-真实类标
d = 100; %降维的维度
max_iter = 5; %迭代次数
alpha1=43; alpha2=36; %LANE模型超参数
delta1 = 0.97; delta2 = 1.6; %计算Htest参数
K = 72; %KNN参数

%留出法 82分
rng(10); sampleidx = randperm(length(Network)); %打乱原数据集
trainidx = sampleidx(1:round(length(sampleidx)*0.8)); %训练集下标
testidx = sampleidx(round(length(sampleidx)*0.8)+1:end); %测试集下标
G = Network(trainidx,trainidx); A = Attributes(trainidx,:); label = Label(trainidx); %训练集
Gtest = Network(testidx,testidx); Atest = Attributes(testidx,:); ltest = Label(testidx); %测试集
clear Attributes Label sampleidx;

%图嵌入学习 -- LANE

%得到LG LA LY
Y = zeros(length(label),length(unique(label)));
for i = 1:length(Y)
    Y(i,label(i)) = 1;
end
YY = Y*Y';
LG = GetLaplacianMartix(G);
LA = GetLaplacianMartix(A);
LY = GetLaplacianMartix(YY);
clear Y i;

%初始化
N = length(label);
UA=zeros(N,d); UY=zeros(N,d); H=zeros(N,d);

%迭代更新
iteration=0;
while true
   [UG,~] = eigs(LG+alpha1*UA*UA'+alpha2*UY*UY'+H*H',d);
   [UA,~] = eigs(alpha1*LA+alpha1*UG*UG'+H*H',d);
   [UY,~] = eigs(alpha2*LY+alpha2*UG*UG'+H*H',d);
   [H,~] = eigs(UG*UG'+UA*UA'+UY*UY',d);
    
   iteration = iteration+1;
   if iteration >= max_iter %终止条件
       break;
   end
end
clear LG LA LY UG UA UY;

%计算Htest
G1 = Network(trainidx,:);
G2 = Network(testidx,:);
Htest = delta1*(G2*pinv(pinv(H)*G1))+delta2*(Atest*pinv(pinv(H)*A));
clear Network G1 G2;

%KNN
prectY = zeros(length(Htest),1);
for i = 1:size(Htest,1)
    prectY(i) = myKNN(Htest(i,:),H,label,K);
end

%评估
classificationACC(ltest,prectY)