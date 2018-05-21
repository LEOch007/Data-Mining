function LM = GetLaplacianMartix(M)

SM = squareform(1-pdist(M,'cosine')); %���ƶȾ���
SM(isnan(SM)) = 0;
DM = eye(size(M,1)); %�Ⱦ���
DM = DM .* sum(SM,2); 
LM = DM^(-.5)* SM *DM^(-.5); %��׼��������˹����

end