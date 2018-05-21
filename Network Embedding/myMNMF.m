clear;
close all;
%����
load('D:\~������\�����ھ�ʵ��\lab3\���ݼ�\texas'); %���� A-�ڽӾ��� F-���������� label-��ʵ���
k = 5; %��������
m = 5; %��ά���ά��
K = 8; %KNN����
lrate = 0.02; %ѧϰ��
imax = 300; %��������
alpha = 0.01; belta = 0.012; lamda = 1; %������

%ͼǶ��ѧϰ 
%M-NMF

%�õ�S����
S1 = A; %һ�����ƶ�
S2 = zeros(length(A),length(A)); %�������ƶ�
for i = 1:length(A)
    for j = 1:i-1
        S2(i,j) = 1-pdist2(A(i,:),A(j,:),'cosine');
        S2(j,i) = S2(i,j);
    end
end
S = S1 + 5*S2; %S����
clear S1 S2;

%�õ�B����
B = zeros(length(A),length(A)); %B����
e = sum(sum(A))/2; %��Ȩ��
for i = 1:length(A)
    for j = 1:length(A)
        ki = sum(A(i,:)); %i�ڵ�Ķ�
        kj = sum(A(j,:));
        B(i,j) = A(i,j) - (ki*kj)/(2*e);
    end
end
clear ki kj e;

%��ʼ��
n = length(A);
rng(2); M = rand(n,m);
rng(3); U = rand(n,m); %n*m�Ǹ�����
rng(5); C = rand(k,m);
rng(6); H = rand(n,k);

%�������� �ݶ��½�
iteration = 0;
while true
    M_temp = lrate* M.* (S*U./ (M*U'*U));
    U_temp = lrate* U.* ((S'*M + alpha*H*C)./ (U*(M'*M + alpha*C'*C)));
    C_temp = lrate* C.* (H'*U./ (C*U'*U));
    
    Delta = (2*belta*(B*H)).* (2*belta*(B*H)) + 16*lamda*(H*H'*H).* (2*belta*A*H + 2*alpha*U*C' + (4*lamda-2*alpha)*H);
    H_temp = H.* sqrt( (-2*belta*B*H + sqrt(Delta))./ (8*lamda*H*H'*H) );
    
    %ͬ������
    M = M_temp;
    U = U_temp;
    C = C_temp;
    H = H_temp;
    
    %��ֹ����
    iteration = iteration+1;
    if iteration >= imax
        break;
    end
end
clear M_temp U_temp H_temp C_temp Delta;

%�������ݼ�
rng(10); randidx = randperm(length(U));  %����ԭ���ݼ�˳��
pos = round(length(randidx)*0.7); %������
trainidx = randidx(1:pos); %ѵ�����±�
prectidx = randidx(pos+1:end); %���Լ��±�

X = U(trainidx,:); Y = label(trainidx); %ѵ��������/��ǩ
prectX = U(prectidx,:); %���Լ�����
valiY = label(prectidx); prectY = zeros(length(prectX),1); %��ʵ��ǩ/Ԥ���ǩ

%KNN
for i = 1:length(prectX)
    prectY(i) = myKNN(prectX(i,:),X,Y,K);
end

%����Ч��
classificationACC(valiY,prectY)