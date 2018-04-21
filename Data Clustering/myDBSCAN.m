clear all;
close all;
X = load('D:\~������\�����ھ�ʵ��\lab1\datasets\Spiral_cluster=3.txt'); %���� --����
C = -1*ones(length(X),1); %��� --���

dis = zeros(length(X),length(X)); %�������
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

%k-����ȷ������
eps = 0; %�뾶
minpts = 0; %�ܶ�
KK = 8;
sdis = sort(dis,2); %����
max_dif = 0;
for k = 1:KK
    kcolumn = sort(sdis(:,k)); %����k-����
    [v,p] = max(diff(kcolumn)); %�������
    if v>max_dif
        max_dif = v;
        eps = kcolumn(p);
        minpts = k;
    end
end

%�Һ��ĵ�
arr = zeros(length(X),length(X)); %ÿ����������
arr(dis<=eps) = 1;
core = find(sum(arr,2)>=minpts); %���ĵ�

%��ǩ����
c=1;
for i=1:length(core)
    if C(core(i)) == -1
        C(core(i)) = c; %��һ�η���
        C(find(arr(core(i),:)==1)) = C(core(i)); 
        c = c+1;
    else
        C(find(arr(core(i),:)==1)) = C(core(i)); %��ǩ����
    end
end
cluster = unique(C);

%������ӻ�����Ч��
% color = 'rbcmkwyg-';
figure;
hold on;
for i=1:length(cluster)
    px = X(C==cluster(i),1);
    py = X(C==cluster(i),2);
    plot(px,py,'.');
end