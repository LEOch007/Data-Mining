%���н��б�׼�� z-score
function X = mynormalize(X)
for i = 1:length(X)
    X(i,:) = ( X(i,:)-mean(X(i,:)) )/ std(X(i,:));
end