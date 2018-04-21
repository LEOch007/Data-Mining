clear all;
close all;
X = load('D:\~大三下\数据挖掘实验\lab1\datasets\Spiral_cluster=3.txt'); %数据 --输入
C = -1*ones(length(X),1); %类标 --输出

dis = zeros(length(X),length(X)); %距离矩阵
for i = 1:length(X)
    for j = 1:i
        if i==j 
            dis(i,j)=0;
        else
            dis(i,j)=sqrt(sum((X(i,:)-X(j,:)).^2,2));
            dis(j,i)=dis(i,j);
        end
    end
end

%k-距离确定参数
eps = 0; %半径
minpts = 0; %密度
KK = 8;
sdis = sort(dis,2); %排序
max_dif = 0;
for k = 1:KK
    kcolumn = sort(sdis(:,k)); %有序k-距离
    [v,p] = max(diff(kcolumn)); %导数最大处
    if v>max_dif
        max_dif = v;
        eps = kcolumn(p);
        minpts = k;
    end
end

%找核心点
arr = zeros(length(X),length(X)); %每个点的领域点
arr(dis<=eps) = 1;
core = find(sum(arr,2)>=minpts); %核心点

%标签传递
c=1;
for i=1:length(core)
    if C(core(i)) == -1
        C(core(i)) = c; %第一次分配
        C(find(arr(core(i),:)==1)) = C(core(i)); 
        c = c+1;
    else
        C(find(arr(core(i),:)==1)) = C(core(i)); %标签传递
    end
end
cluster = unique(C);

%输出可视化聚类效果
% color = 'rbcmkwyg-';
figure;
hold on;
for i=1:length(cluster)
    px = X(C==cluster(i),1);
    py = X(C==cluster(i),2);
    plot(px,py,'.');
end