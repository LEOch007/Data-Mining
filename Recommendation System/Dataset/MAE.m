%Æ½¾ù¾ø¶ÔÎó²î
function rate = MAE(prect,actual)
    rate = sum(abs(prect-actual))/length(actual);
end