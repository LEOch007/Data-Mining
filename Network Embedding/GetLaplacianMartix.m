function LM = GetLaplacianMartix(M)

SM = squareform(1-pdist(M,'cosine')); %相似度矩阵
SM(isnan(SM)) = 0;
DM = eye(size(M,1)); %度矩阵
DM = DM .* sum(SM,2); 
LM = DM^(-.5)* SM *DM^(-.5); %标准化拉普拉斯矩阵

end