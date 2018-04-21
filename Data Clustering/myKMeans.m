clear all;
%二维可视化数据集
X = load('D:\~大三下\数据挖掘实验\lab1\datasets\Aggregation_cluster=7.txt'); %数据 --输入
K = 7; %类数
C = zeros(length(X),1); %类标 --输出
Center = zeros(K,2); %质心
Cennum = zeros(K,1); %类中数据量

%初始化K个质心
Centeridx = zeros(K,1);
rand('seed',10); %实验的可重复性
first = randi([1,length(X)]); %先随机找第一个质心
Centeridx(1) = first;
for i = 2:K %尽可能选远的K-1个
    distance = zeros(length(X),1);
    for j = 1:length(X)
        distance(j) = min( sum(abs(X(j,:)-X(Centeridx(1:i-1),:)),2) ); %曼哈顿距离最小值
    end
    distance = distance/sum(distance); %归一化距离
    temp = 0;
    while(ismember(temp,Centeridx)) %剔除重复质心
        temp = randsrc(1,1,[1:length(X);distance']); %按概率抽取
    end 
    Centeridx(i)=temp;
end
Center = X(Centeridx,:);

iteration = 1;
while(iteration<200)
    %分配类标
    Cennum(:,:) = 0;
    for i = 1:length(X)
        [v,index]=min(sum((X(i,:)-Center).^2,2)); %欧式距离
        C(i)=index;
        Cennum(index) = Cennum(index)+1;
    end

    %重新计算质心
    for k = 1:K
        Center(k,:) = sum(X(C==k,:),1);
    end
    Center = Center./Cennum;
    
    iteration = iteration+1;
end

%输出可视化聚类效果
% color = 'rmgcbwyk-';
figure;
hold on;
for k=1:K
    px = X(C==k,1);
    py = X(C==k,2);
    plot(px,py,'.');
end