clear all;
close all;
%输入
load('D:\~大三下\数据挖掘实验\lab2\dataset\texas.mat'); %数据 A-邻接矩阵 F-特征向量表 label-真实类标
sigma = 0.078; %高斯核函数的超参数
%输出
C = 1:length(A); %类标 初始化为各自不同社区

%计算距离矩阵
W = zeros(length(A),length(A));
for i = 1:length(A)
    for j = 1:i-1
        W(i,j) = exp( - pdist2(F(i,:),F(j,:),'cosine') / (2*sigma*sigma) ); %高斯核函数
        W(j,i) = W(i,j);
    end
end

% Louvain
m = sum(A(:))/2; %整个网络边权之和
past_m = -2; %极小旧模块度
while true
    community = unique(C); %当前社区
    rng('default'); %保证实验可重复性
    community = community( randperm(length(community)) ); %随机打乱顺序
    
    %遍历每个社区 判断是否合并
    for i = 1:length(community)
        neighbor= zeros(1,length(A)); %邻居
        communitymember = find(C==community(i)); %社区成员

        for j = 1:length(communitymember)
            neighbor( A(communitymember(j),:)==1 )=1; %找所有邻居
        end
        neighbor(communitymember)=0; %除去社区成员
        
        cluster = unique(C(neighbor==1)); %邻居的社群
        rng('default'); %保证实验可重复性
        cluster = cluster( randperm(length(cluster)) ); %随机打乱顺序

        old_modularity = modular(C,W,m); %原模块度
        max = 0; max_c = -1;
        %遍历各个相邻社群
        for j = 1:length(cluster) 
            Ccopy = C;
            Ccopy(communitymember) = cluster(j); %假设合并
            new_modularity = modular(Ccopy,W,m); %新模块度
            delta = new_modularity - old_modularity; %模块度增益
            
            %记录增益最大的
            if delta > max 
                max = delta;
                max_c = cluster(j);
            end
        end
        
        %有增益 加入该社区
        if max > 0 
            C(communitymember) = max_c;
        end
    end
    
    %终止条件
    current_m = modular(C,W,m);
    if current_m <= past_m %模块度不再增加
        break;
    end
    past_m = current_m;
end

ClusteringMeasure(label,C) %评估