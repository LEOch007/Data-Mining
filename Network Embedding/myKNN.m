function result = myKNN(prect,trainX,trainY,K)
%
% prect一个测试样本；trainX训练样本,trainY训练样本对应的标签,K参数
%
diffMat = abs( repmat(prect,[size(trainX,1),1]) - trainX );
distanceMat = sum(diffMat,2);
[B,IX] = sort(distanceMat,'ascend');
len = min(K,length(B));
result = mode(trainY(IX(1:len)));

end