
function ac = classificationACC(label,preY)
    n = length(label);
    
    x = (label - preY);
    
    num = length(find(x == 0));
    ac = num/n;


end