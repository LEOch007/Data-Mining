clear all;
close all;
%����
load('D:\~������\�����ھ�ʵ��\lab2\dataset\texas.mat'); %���� A-�ڽӾ��� F-���������� label-��ʵ���
sigma = 0.078; %��˹�˺����ĳ�����
%���
C = 1:length(A); %��� ��ʼ��Ϊ���Բ�ͬ����

%����������
W = zeros(length(A),length(A));
for i = 1:length(A)
    for j = 1:i-1
        W(i,j) = exp( - pdist2(F(i,:),F(j,:),'cosine') / (2*sigma*sigma) ); %��˹�˺���
        W(j,i) = W(i,j);
    end
end

% Louvain
m = sum(A(:))/2; %���������Ȩ֮��
past_m = -2; %��С��ģ���
while true
    community = unique(C); %��ǰ����
    rng('default'); %��֤ʵ����ظ���
    community = community( randperm(length(community)) ); %�������˳��
    
    %����ÿ������ �ж��Ƿ�ϲ�
    for i = 1:length(community)
        neighbor= zeros(1,length(A)); %�ھ�
        communitymember = find(C==community(i)); %������Ա

        for j = 1:length(communitymember)
            neighbor( A(communitymember(j),:)==1 )=1; %�������ھ�
        end
        neighbor(communitymember)=0; %��ȥ������Ա
        
        cluster = unique(C(neighbor==1)); %�ھӵ���Ⱥ
        rng('default'); %��֤ʵ����ظ���
        cluster = cluster( randperm(length(cluster)) ); %�������˳��

        old_modularity = modular(C,W,m); %ԭģ���
        max = 0; max_c = -1;
        %��������������Ⱥ
        for j = 1:length(cluster) 
            Ccopy = C;
            Ccopy(communitymember) = cluster(j); %����ϲ�
            new_modularity = modular(Ccopy,W,m); %��ģ���
            delta = new_modularity - old_modularity; %ģ�������
            
            %��¼��������
            if delta > max 
                max = delta;
                max_c = cluster(j);
            end
        end
        
        %������ ���������
        if max > 0 
            C(communitymember) = max_c;
        end
    end
    
    %��ֹ����
    current_m = modular(C,W,m);
    if current_m <= past_m %ģ��Ȳ�������
        break;
    end
    past_m = current_m;
end

ClusteringMeasure(label,C) %����