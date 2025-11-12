clear; clc; rng(0);

load("features_dataset.mat");  

% Remove non-feature columns
predictorNames = string(features.Properties.VariableNames);
predictorNames = predictorNames(~ismember(predictorNames, ["User","Day","WindowID"]));

X_tbl = features(:, predictorNames);  
X = table2array(X_tbl);              
Y = features.User;                   

if any(isnan(X(:)))
    warning('X contains NaNs - replacing NaNs with column median.');
    % Replace NaN by column median
    for c = 1:size(X,2)
        col = X(:,c);
        col(isnan(col)) = median(col(~isnan(col)));
        X(:,c) = col;
    end
end

% Z-score 

mu = mean(X,1);
sigma = std(X,[],1);
sigma_fixed = sigma;
sigma_fixed(sigma_fixed == 0) = 1;  
X = (X - mu) ./ sigma_fixed;

% Stratified 80/20 split. Use cvpartition for stratified split
rng(0);
cv = cvpartition(Y, 'HoldOut', 0.20);
trainMask = training(cv);
testMask  = test(cv);

Xtrain_raw = X(trainMask, :);   
Ytrain_raw = Y(trainMask);
Xtest_raw  = X(testMask, :);    
Ytest = Y(testMask);

% Transpose for neural net toolbox: features x samples
Xtrain = Xtrain_raw';
Xtest  = Xtest_raw';

% One-hot targets 
numClasses = numel(unique(Y));
Ttrain = full(ind2vec(Ytrain_raw'));   
Ttest  = full(ind2vec(Ytest'));       

% Build & tune MLP
hiddenCandidates = [20, 50, 100];   
bestValAcc = -inf;
bestNet = [];
bestHidden = NaN;
useInternalValidation = true;  

for h = hiddenCandidates
    rng(0); 
    net = patternnet(h, 'trainscg');  
    
    if useInternalValidation
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio   = 15/100;
        net.divideParam.testRatio  = 15/100;
    else
        net.divideParam.trainRatio = 1.0;
        net.divideParam.valRatio = 0.0;
        net.divideParam.testRatio = 0.0;
    end

    % Training params
    net.trainParam.epochs = 300;
    net.trainParam.max_fail = 12;  
    net.performFcn = 'crossentropy';

    [netTrained, tr] = train(net, Xtrain, Ttrain);

    % Extract validation indices from 'tr' if available
    if useInternalValidation && isfield(tr, 'vind') && ~isempty(tr.vind)
        valIdx = tr.vind;   
        Xval = Xtrain(:, valIdx);
        Tval = Ttrain(:, valIdx);
        YvalOut = netTrained(Xval);
        [~, yvalpred] = max(YvalOut, [], 1);
        yvalpred = yvalpred';
        YvalTrue = vec2ind(Tval)'; 
        valAcc = mean(yvalpred == YvalTrue) * 100;
    else
        YvalOut = netTrained(Xtrain);
        [~, yvalpred] = max(YvalOut, [], 1);
        yvalpred = yvalpred';
        YvalTrue = Ytrain_raw;
        valAcc = mean(yvalpred == YvalTrue) * 100;
    end

    fprintf('Hidden %d -> val acc = %.2f%%\n', h, valAcc);

    if valAcc > bestValAcc
        bestValAcc = valAcc;
        bestNet = netTrained;
        bestHidden = h;
    end
end

% Use bestNet
net = bestNet;
hiddenUnits = bestHidden;
fprintf('Selected hidden units = %d (val acc = %.2f%%)\n', hiddenUnits, bestValAcc);

% Final prediction on external test set
Yhat_proba = net(Xtest);                   
[~, ypred] = max(Yhat_proba, [], 1);       
ypred = ypred';                            

% Accuracy
acc = mean(ypred == Ytest) * 100;
fprintf('Final MLP Accuracy (external test 20%%): %.2f%%\n', acc);

% Confusion matrix (numeric)
C = confusionmat(Ytest, ypred);
disp('Confusion matrix (rows=true, cols=pred):');
disp(C);

% Per-class precision, recall
precision = diag(C) ./ (sum(C,1)' + eps);
recall = diag(C) ./ (sum(C,2) + eps);
F1 = 2 .* (precision .* recall) ./ (precision + recall + eps);

Tclass = table((1:numel(precision))', precision, recall, F1, ...
    'VariableNames', {'Class','Precision','Recall','F1'});
disp(Tclass);

% Top-3 accuracy 
[~, sortedIdx] = sort(Yhat_proba, 1, 'descend'); 
top3 = sortedIdx(1:min(3,size(sortedIdx,1)), :);  
top3acc = mean(any(top3' == Ytest, 2)) * 100;
fprintf('Top-3 accuracy is %.2f%%\n', top3acc);

% Plot confusion matrix
figure;
plotconfusion(Ttest, Yhat_proba);
title(sprintf('MLP Confusion Matrix (hidden=%d) — Acc %.2f%%', hiddenUnits, acc));
saveas(gcf, 'MLP_confusion_randomsplit.png');

classLabels = unique(Y); 
save("mlp_randomsplit.mat", "net", "mu", "sigma_fixed", "predictorNames", ...
    "trainMask", "testMask", "Ytest", "Xtest", "Ttest", "hiddenUnits", "bestValAcc", "classLabels");

fprintf('Model and metadata saved to mlp_randomsplit.mat\n');