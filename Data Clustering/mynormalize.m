%按行进行标准化 z-score
function X = mynormalize(X)
for i = 1:length(X)
    X(i,:) = ( X(i,:)-mean(X(i,:)) )/ std(X(i,:));
end