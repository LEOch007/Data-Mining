%ƽ���������
function rate = MAE(prect,actual)
    rate = sum(abs(prect-actual))/length(actual);
end