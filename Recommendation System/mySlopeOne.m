%���ݼ�1
temp = load('D:\~������\�����ھ�ʵ��\lab4\Lab4_���ݼ�\movielens���ݼ�\ml-100k\ml-100k\u.data');
data = temp(:,1:3); 
clear temp;
%�������ݼ�
data_train = data(1:round(length(data)*0.8),:); %ѵ����
data_test = data(round(length(data)*0.8)+1:end,:); %���Լ�

%���� �û�-��Ʒ���־���
num_u = max(data(:,1));
num_i = max(data(:,2));
R = -1*ones(num_u,num_i);
for i = 1:size(data_train,1)
    R(data_train(i,1),data_train(i,2)) = data_train(i,3);
end

%���� ����ƫ�����
dev = zeros(num_i,num_i);
for i = 1:num_i
    for j = 1:i-1
        same = intersect(find(R(:,i)~=-1),find(R(:,j)~=-1)); %ͬʱ���ֵ��û�
        if isempty(same)
            %û��ͬʱ����������Ʒ���ֵ��û�
            dev(i,j) = -1000;
        else
            %��������ƫ��
            dev(i,j) = sum(R(same,i)-R(same,j))/length(same);
        end
        dev(j,i) = dev(i,j); %---
    end
end
            
%Ԥ���û�δ������Ʒ
mae = -1*ones(num_u,1);
rmse = -1*ones(num_u,1);
for i = 1:num_u
    goods = data_test(data_test(:,1)==i,2); %�û�i��������Ʒ
    if isempty(goods)
        continue;
    end
    rgood = find(R(i,:)~=-1); %�û�i�����ֵ���Ʒ
    score = -1*ones(length(goods),1); %Ԥ��ķ���
    for j = 1:length(goods)
        sgood = intersect(rgood,find(dev(goods(j),:)~=-1000)); %����Ʒgoods(j)������ƫ�����Ʒ
        if isempty(sgood) %�� 
            score(j) = mean(R(i,rgood)); %ȡ�û����־�ֵ
        else 
            score(j) = sum(dev(goods(j),sgood)+R(i,sgood))/length(sgood); %ƫ��ȡֵ
        end
    end
    
    %���Լ�
    actual = data_test(data_test(:,1)==i,3); %ʵ������
    mae(i) = MAE(score,actual); %MAE���
    rmse(i) = RMSE(score,actual); %RMSE���
end

%�������
m_mae = mean(mae);
m_rmse = mean(rmse);
