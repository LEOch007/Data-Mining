clear all;
close all;
%����
load('D:\~������\�����ھ�ʵ��\lab2\dataset\texas.mat'); %���� A-�ڽӾ��� F-���������� label-��ʵ���
K = 5; %��ĸ���
KK = 8; %KmeansԤ���������
sigma = 0.4; %��˹�˺����ĳ�����

%���
C = -1*ones(length(A),1); %���

%�������ƶȾ���
W = zeros(length(A),length(A));
for i = 1:length(A)
    for j = 1:i-1
%         W(i,j) = 1-pdist2(F(i,:),F(j,:),'cosine'); %���Ҿ���
        W(i,j) = exp( - pdist2(F(i,:),F(j,:),'cosine') / (2*sigma*sigma) ); %��˹�˺���
        W(j,i) = W(i,j);
    end
end

%�����Ⱦ���
D = eye(length(A));
D = D .* sum(W,2); %�������ƶȼ���Ⱦ���

%�õ���׼��������˹����
L = D-W;
L = D^(-.5)*L*D^(-.5);

%�õ���������
[eigVector,eigvalue] = eig(L); %������������ & ����ֵ����
[eigvalue,index] = sort(diag(eigvalue));
FM = eigVector(:,index(2:KK+1));
FM = mynormalize(FM); %���б�׼��

%������������о���
rng('default'); %��֤ʵ����ظ���
CC = kmeans(FM,KK); % Ĭ�Ͼ��� -ŷʽ Ĭ�ϳ�ʼ��ʽ -Kmeans++ Ĭ������������ -100

%�ϲ�����
while KK~=K
    Nmin = 100000000; combine = [-1, -1];
    
    %���кϲ�������
    for i = 1:KK
        for j = 1:i-1
            Ccopy = CC;
            Ccopy(Ccopy==i) = j; %��ϲ�
            Ccluster = unique(Ccopy); %�ϲ�����������

            Ncut = 0; %�ϲ����Ncut
            for l = 1:length(Ccluster) %�����ϲ�����������
                Ncut = Ncut + ( sum(sum(W(Ccopy==Ccluster(l),Ccopy~=Ccluster(l)))) /  sum(sum(W(Ccopy==Ccluster(l),:))) );
            end
            if Ncut<Nmin %��С��Ncut�ĺϲ���Ϸ�ʽ
                Nmin = Ncut;
                combine = [i,j];
            end
        end
    end
    CC(CC==combine(1)) = combine(2); %��ʵ�ϲ�
    KK = KK-1;
end

%���۾���Ч��
ClusteringMeasure(label,CC)