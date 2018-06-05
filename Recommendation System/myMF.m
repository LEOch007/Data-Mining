%���ݼ�1
temp = load('D:\~������\�����ھ�ʵ��\lab4\Lab4_���ݼ�\movielens���ݼ�\ml-100k\ml-100k\u.data');
data = temp(:,1:3); 
clear temp;
%�������ݼ�
data_train = data(1:round(length(data)*0.8),:); %ѵ����
data_test = data(round(length(data)*0.8)+1:end,:); %���Լ�

%����
d = 30; %ά��
maxIter = 20; %��������
alpha = 0.01;
lambda = 10;

%��ʼ��UV
num_u = max(data(:,1));
num_i = max(data(:,2));
rng(10); U = rand(num_u,d);
rng(10); V = rand(num_i,d);
U_last = U;
V_last = V;

%����ݶ��½�������UV
for limit = 1:maxIter
    %�������ݼ�
    for i = 1:size(data_train,1)
        tempUser = data_train(i,1);  %�û�
        tempItem = data_train(i,2);  %��Ʒ
        tempGrade = data_train(i,3); %��ʵ����
        E = tempGrade - U(tempUser,:) * V(tempItem,:)'; %��ֵ
        U(tempUser,:) = U_last(tempUser,:) + alpha * (E*V_last(tempItem,:) - lambda*U_last(tempUser,:)); %����
        V(tempItem,:) = V_last(tempItem,:) + alpha * (E*U_last(tempUser,:) - lambda*V_last(tempItem,:)); %����
    end
    U_last = U;
    V_last = V;
    alpha = alpha/(1+limit); %��̬ѧϰ��
end

%Ԥ����
R_prect = U*V';
actual = data_test(:,3);
prect = -1*ones(size(data_test,1),1);
for l = 1:size(data_test,1)
    prect(l) = R_prect(data_test(l,1),data_test(l,2));
end

%����
mae = MAE(prect,actual)
rmse = RMSE(prect,actual)