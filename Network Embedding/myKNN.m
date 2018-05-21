function result = myKNN(prect,trainX,trainY,K)
%
% prectһ������������trainXѵ������,trainYѵ��������Ӧ�ı�ǩ,K����
%
diffMat = abs( repmat(prect,[size(trainX,1),1]) - trainX );
distanceMat = sum(diffMat,2);
[B,IX] = sort(distanceMat,'ascend');
len = min(K,length(B));
result = mode(trainY(IX(1:len)));

end