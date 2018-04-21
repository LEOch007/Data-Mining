clear all;
%��ά���ӻ����ݼ�
X = load('D:\~������\�����ھ�ʵ��\lab1\datasets\Aggregation_cluster=7.txt'); %���� --����
K = 7; %����
C = zeros(length(X),1); %��� --���
Center = zeros(K,2); %����
Cennum = zeros(K,1); %����������

%��ʼ��K������
Centeridx = zeros(K,1);
rand('seed',10); %ʵ��Ŀ��ظ���
first = randi([1,length(X)]); %������ҵ�һ������
Centeridx(1) = first;
for i = 2:K %������ѡԶ��K-1��
    distance = zeros(length(X),1);
    for j = 1:length(X)
        distance(j) = min( sum(abs(X(j,:)-X(Centeridx(1:i-1),:)),2) ); %�����پ�����Сֵ
    end
    distance = distance/sum(distance); %��һ������
    temp = 0;
    while(ismember(temp,Centeridx)) %�޳��ظ�����
        temp = randsrc(1,1,[1:length(X);distance']); %�����ʳ�ȡ
    end 
    Centeridx(i)=temp;
end
Center = X(Centeridx,:);

iteration = 1;
while(iteration<200)
    %�������
    Cennum(:,:) = 0;
    for i = 1:length(X)
        [v,index]=min(sum((X(i,:)-Center).^2,2)); %ŷʽ����
        C(i)=index;
        Cennum(index) = Cennum(index)+1;
    end

    %���¼�������
    for k = 1:K
        Center(k,:) = sum(X(C==k,:),1);
    end
    Center = Center./Cennum;
    
    iteration = iteration+1;
end

%������ӻ�����Ч��
% color = 'rmgcbwyk-';
figure;
hold on;
for k=1:K
    px = X(C==k,1);
    py = X(C==k,2);
    plot(px,py,'.');
end