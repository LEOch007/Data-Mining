clear all;
close all;
%输入
X = load('D:\~大三下\数据挖掘实验\lab1\datasets\Pathbased_cluster=3.txt'); %数据 
percent = 0.02; %对dc的选取比例
K = 3;
%输出
C = -1*ones(length(X),1); %类标

%计算距离矩阵
dis = zeros(length(X),length(X)); %距离矩阵
disvec = zeros( length(X)*(length(X)-1)/2 ,1); %距离向量
k = 1;
for i = 1:length(X)
    for j = 1:i-1
        dis(i,j)=sqrt(sum( (X(i,:)-X(j,:)).^2,2 )); %欧式距离
        dis(j,i)=dis(i,j);
        disvec(k) = dis(i,j);
        k = k+1;
    end
end

%选取参数dc
disvec = sort(disvec); %升序排列
dc = disvec( round(percent*length(disvec)) ); %选取邻域参数dc

%计算密度向量p
p = zeros(length(X),1);
for i = 1:length(X)
    p(i) =  sum( exp(-(dis(i,:)/dc).^2),2 ) - exp(-(dis(i,i)/dc).^2); %高斯核函数 除去自身
end
[psort,pidx] = sort(p,'descend'); %降序排列 密度大到小
[pmax,midx] = max(p); %最大密度点

%计算距离向量d
d = zeros(length(X),1);
neighbor = zeros(length(X),1); % arg(d)代表邻居
for i = 1:length(X)
    if p(i)==pmax
        [d(i),neighbor(i)] = max( dis(i,:) );
    else
        idxset = find(p>p(i)); %密度大
        d(i) = min( dis(i,idxset) ); %数组经过截取 第二维度大小发生了变化 返回的下标并非原始数据集下标
        neighs = intersect(find(dis(i,:)==d(i)),idxset);
        neighbor(i) = neighs(1);
    end
end

%计算乘积r 找到类中心

r = p.*d;
[rsort,ridx] = sort(r,'descend'); %降序排列
center = ridx(1:K); %类中心

%类中心分配类标
for k = 1:K
    C(center(k))=k; 
end

%非类中心分配类标
%按序传递法
for i = 1:length(X)
    if C(pidx(i)) == -1
        C(pidx(i)) = C(neighbor(pidx(i)));
    end
end
cluster = unique(C);

%输出可视化聚类效果
figure;
hold on;
for i=1:length(cluster)
    px = X(C==cluster(i),1);
    py = X(C==cluster(i),2);
    plot(px,py,'.');
end