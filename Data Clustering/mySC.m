clear all;
close all;
%����
X = load('D:\~������\�����ھ�ʵ��\lab1\datasets\Aggregation_cluster=7.txt'); %���� 
sigma = 0.29; %��˹�˺����ĳ�����
K = 7; %��ĸ���
% rng('shuffle');
% sigma = 0.1 + (10-0.1).*rand(1); %���ѡȡ����

%���
C = -1*ones(length(X),1); %���

%�������ƶȾ���
W = zeros(length(X),length(X)); %������������ƶ�
for i = 1:length(X)
    for j = 1:i-1
        W(i,j) = exp( - sum((X(i,:)-X(j,:)).*(X(i,:)-X(j,:))) / (2*sigma*sigma) ); %��˹�˺���
        W(j,i) = W(i,j); %�Գƾ���
    end
end

%�����Ⱦ���
D = eye(length(X));
D = D .* sum(W,2); %�������ƶȼ���Ⱦ���

%�õ���׼��������˹����
L = D-W;
L = D^(-.5)*L*D^(-.5);

%�õ���������
[eigVector,eigvalue] = eig(L); %������������ & ����ֵ����
[eigvalue,index] = sort(diag(eigvalue));
F = eigVector(:,index(2:K+1));
F = mynormalize(F);

%������������о���
rng('default'); %��֤ʵ����ظ���
C = kmeans(F,K); % Ĭ�Ͼ��� -ŷʽ Ĭ�ϳ�ʼ��ʽ -Kmeans++ Ĭ������������ -100

%������ӻ�����Ч��
figure;
hold on;
for k=1:K
    px = X(C==k,1);
    py = X(C==k,2);
    plot(px,py,'.');
end