function [FP, FN, TP, TN, acc, prec, rec, f_meas, TPR, FPR]=performance(likelihoods, labels, thresh)
    FP = 0;
    FN = 0;
    TP = 0;
    TN = 0;
    acc = 0;
    prec = 0;
    rec = 0;
    f_meas = 0;
    TPR = 0;
    FPR = 0;
    for i=1:length(likelihoods)
        if likelihoods(i) < thresh
            if labels(i) == 1
                FN = FN + 1;
            else
                TN = TN + 1;
            end
        elseif likelihoods(i) >= thresh
            if labels(i) == -1
                FP = FP + 1;
            else
                TP = TP + 1;
            end
        end
    end
    
    acc = (TP + TN)/(TP + TN + FN + FP);
    prec = TP/(TP+FP);
    rec = TP/(TP+FN);
    f_meas = 2*prec*rec/(prec + rec);
    
    TPR = TP/(TP+FN);
    FPR = FP/(FP+TN);
end