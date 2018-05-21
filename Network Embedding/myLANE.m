clear;
close all;
%����
load('D:\~������\�����ھ�ʵ��\lab3\���ݼ�\texas'); %���� Network-�ڽӾ��� Attributes-���������� Label-��ʵ���
d = 100; %��ά��ά��
max_iter = 5; %��������
alpha1=43; alpha2=36; %LANEģ�ͳ�����
delta1 = 0.97; delta2 = 1.6; %����Htest����
K = 72; %KNN����

%������ 82��
rng(10); sampleidx = randperm(length(Network)); %����ԭ���ݼ�
trainidx = sampleidx(1:round(length(sampleidx)*0.8)); %ѵ�����±�
testidx = sampleidx(round(length(sampleidx)*0.8)+1:end); %���Լ��±�
G = Network(trainidx,trainidx); A = Attributes(trainidx,:); label = Label(trainidx); %ѵ����
Gtest = Network(testidx,testidx); Atest = Attributes(testidx,:); ltest = Label(testidx); %���Լ�
clear Attributes Label sampleidx;

%ͼǶ��ѧϰ -- LANE

%�õ�LG LA LY
Y = zeros(length(label),length(unique(label)));
for i = 1:length(Y)
    Y(i,label(i)) = 1;
end
YY = Y*Y';
LG = GetLaplacianMartix(G);
LA = GetLaplacianMartix(A);
LY = GetLaplacianMartix(YY);
clear Y i;

%��ʼ��
N = length(label);
UA=zeros(N,d); UY=zeros(N,d); H=zeros(N,d);

%��������
iteration=0;
while true
   [UG,~] = eigs(LG+alpha1*UA*UA'+alpha2*UY*UY'+H*H',d);
   [UA,~] = eigs(alpha1*LA+alpha1*UG*UG'+H*H',d);
   [UY,~] = eigs(alpha2*LY+alpha2*UG*UG'+H*H',d);
   [H,~] = eigs(UG*UG'+UA*UA'+UY*UY',d);
    
   iteration = iteration+1;
   if iteration >= max_iter %��ֹ����
       break;
   end
end
clear LG LA LY UG UA UY;

%����Htest
G1 = Network(trainidx,:);
G2 = Network(testidx,:);
Htest = delta1*(G2*pinv(pinv(H)*G1))+delta2*(Atest*pinv(pinv(H)*A));
clear Network G1 G2;

%KNN
prectY = zeros(length(Htest),1);
for i = 1:size(Htest,1)
    prectY(i) = myKNN(Htest(i,:),H,label,K);
end

%����
classificationACC(ltest,prectY)