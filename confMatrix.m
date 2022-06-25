function [conf_matrix, conf_table] = confMatrix(test_labels, pred_labels)

    global anomaly_class;
    
    % ( TP   FN )
    % ( FP   TN )

    % Confusion matrix
    conf_matrix = zeros(2,2);
    for sample=1:length(pred_labels)
        if test_labels(sample) == pred_labels(sample)
            if test_labels(sample) == anomaly_class
                i = 1; j = i;
            else
                i = 2; j = i;
            end
        else
            if test_labels(sample) == anomaly_class
                i = 2; j = 1;
            else
                i = 1; j = 2;
            end
        end

        conf_matrix(i,j) = conf_matrix(i,j) + 1;
    end

    conf_table = array2table(conf_matrix,'RowNames',{'Positive';'Negative'},'VariableNames',{'Positive';'Negative'});
end