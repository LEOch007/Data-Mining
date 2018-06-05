% %С���ݼ�
% data = load ('D:\~������\�����ھ�ʵ��\lab4\Lab4_���ݼ�\��С�����ݼ�\train_small.txt');
% %С���ݼ�2
% data = load ('D:\~������\�����ھ�ʵ��\lab4\Lab4_���ݼ�\��С�����ݼ�\train_small_2.txt'); 
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

%���� ��Ʒ���ƶȾ���
S = ones(num_i,num_i);
for i = 1:num_i
    for j = 1:i-1
        same = intersect(find(R(:,i)~=-1),find(R(:,j)~=-1)); %ͬʱ���ֵ��û�
        if isempty(same)
            %û��ͬʱ����������Ʒ���ֵ��û�
            S(i,j) = 0;
        else
            %Ƥ��ѷ���ϵ��
            ri = mean(R(same,i)); rj = mean(R(same,j));
            fz = sum( (R(same,i)-ri).*(R(same,j)-rj) );
            if fz==0
                S(i,j) = 0; %��ֹNaN�ĳ���
            else
                fm = sqrt(sum( (R(same,i)-ri).^2 )) * sqrt(sum( (R(same,j)-rj).^2 ));
                S(i,j) = fz / fm;
            end
        end
        S(j,i) = S(i,j);
    end
end

%Ԥ���û�δ������Ʒ
N = 20;
alpha = 0.005; %ƽ������
mae = -1*ones(num_u,1);
rmse = -1*ones(num_u,1);
for i = 1:num_u
    goods = data_test(data_test(:,1)==i,2); %δ������Ʒ
    if isempty(goods)
        continue;
    end
    rgood = find(R(i,:)~=-1); %������Ʒ
    score = -1*ones(length(goods),1); %Ԥ��ķ���
    for j = 1:length(goods)
        %�ҽ���
        Neighbors = zeros(N,1); %����
        count = 1; %��������
        [~,id] = sort(S(goods(j),:),'descend'); %��������
        for l=1:length(id)
            if count > N %������������
                break;
            end
            if ismember(id(l),rgood) %���������
                Neighbors(count) = id(l);
                count = count+1;
            end
        end

        Rating = R(i,Neighbors(1:count-1)); %������Ʒ����
        Similar = S(goods(j),Neighbors(1:count-1)); %���ƶ�
        score(j) = (sum(Similar.*Rating)+alpha)/(sum(Similar)+(count-1)*alpha); %Ԥ��ķ���
    end
    
    %���Լ�
    test = data_test(data_test(:,1)==i,:);
    actual = test(:,3); %ʵ������
    prect = score; %Ԥ������
    
    mae(i) = MAE(prect,actual); %MAE���
    rmse(i) = RMSE(prect,actual); %RMSE���
end

%�������
m_mae = mean(mae);
m_rmse = mean(rmse);
        