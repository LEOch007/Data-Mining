%�������
function rate = RMSE(prect,actual)
    rate = sqrt(sum((prect-actual).*(prect-actual))/length(actual));
end