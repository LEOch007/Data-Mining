clear all;
close all;
%����
X = load('D:\~������\�����ھ�ʵ��\lab1\datasets\Pathbased_cluster=3.txt'); %���� 
percent = 0.02; %��dc��ѡȡ����
K = 3;
%���
C = -1*ones(length(X),1); %���

%����������
dis = zeros(length(X),length(X)); %�������
disvec = zeros( length(X)*(length(X)-1)/2 ,1); %��������
k = 1;
for i = 1:length(X)
    for j = 1:i-1
        dis(i,j)=sqrt(sum( (X(i,:)-X(j,:)).^2,2 )); %ŷʽ����
        dis(j,i)=dis(i,j);
        disvec(k) = dis(i,j);
        k = k+1;
    end
end

%ѡȡ����dc
disvec = sort(disvec); %��������
dc = disvec( round(percent*length(disvec)) ); %ѡȡ�������dc

%�����ܶ�����p
p = zeros(length(X),1);
for i = 1:length(X)
    p(i) =  sum( exp(-(dis(i,:)/dc).^2),2 ) - exp(-(dis(i,i)/dc).^2); %��˹�˺��� ��ȥ����
end
[psort,pidx] = sort(p,'descend'); %�������� �ܶȴ�С
[pmax,midx] = max(p); %����ܶȵ�

%�����������d
d = zeros(length(X),1);
neighbor = zeros(length(X),1); % arg(d)�����ھ�
for i = 1:length(X)
    if p(i)==pmax
        [d(i),neighbor(i)] = max( dis(i,:) );
    else
        idxset = find(p>p(i)); %�ܶȴ�
        d(i) = min( dis(i,idxset) ); %���龭����ȡ �ڶ�ά�ȴ�С�����˱仯 ���ص��±겢��ԭʼ���ݼ��±�
        neighs = intersect(find(dis(i,:)==d(i)),idxset);
        neighbor(i) = neighs(1);
    end
end

%����˻�r �ҵ�������

r = p.*d;
[rsort,ridx] = sort(r,'descend'); %��������
center = ridx(1:K); %������

%�����ķ������
for k = 1:K
    C(center(k))=k; 
end

%�������ķ������
%���򴫵ݷ�
for i = 1:length(X)
    if C(pidx(i)) == -1
        C(pidx(i)) = C(neighbor(pidx(i)));
    end
end
cluster = unique(C);

%������ӻ�����Ч��
figure;
hold on;
for i=1:length(cluster)
    px = X(C==cluster(i),1);
    py = X(C==cluster(i),2);
    plot(px,py,'.');
end