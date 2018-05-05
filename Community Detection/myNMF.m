clear;
close all;
%����
load('D:\~������\�����ھ�ʵ��\lab2\dataset\washington.mat'); %���� A-�ڽӾ��� F-���������� label-��ʵ���
K = 5; %��ĸ���
alpha = 0.06; %����
%���
C = 1:length(A); %��� ��ʼ��Ϊ���Բ�ͬ����

%�����ʼ��UV����
N = length(A); %���ݸ���
rng(10); %��֤ʵ����ظ���
U = rand(N,K); %N*K�Ǹ�����
rng(3);
V = rand(N,K);

%��������UV����
iteration = 0;
while true
    U_temp = U; V_temp = V; %��ֹͬ������
    AV = A*V; UVV = U*V'*V;
    AU = A'*U; VUU = V*U'*U;
    %U
    for i = 1:N
        for j = 1:K
            U_temp(i,j) = U(i,j)*AV(i,j)/UVV(i,j); %�첽����
        end
    end
    %V
    for i = 1:N
        for j = 1:K
            V_temp(i,j) = V(i,j)*AU(i,j)/VUU(i,j); %�첽����
        end
    end
    %����UV
    U = U_temp; V = V_temp; 
    iteration = iteration+1;
    
    %��ֹ����
    if iteration >= 100
        break;
    end
end

%�������
% [mvalue,midx] = max(U,[],2); %Uÿһ�����ֵ
% C = midx; %�������
C = kmeans(U,K);

ClusteringMeasure(label,C) %�������