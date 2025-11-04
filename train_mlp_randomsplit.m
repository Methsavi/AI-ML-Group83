%% train_mlp_randomsplit.m
% Revised training script for MLP with stratified 80/20 split, robust z-score,
% optional internal validation, small hyperparameter sweep, and detailed evaluation.
clear; clc; rng(0);

% -------------------------
% 1) Load data
% -------------------------
load("features_dataset.mat");   % expects 'features' table
% Inspect: uncomment to check
% summary(features)
% head(features)

% -------------------------
% 2) Prepare predictors & labels
% -------------------------
% Remove non-feature columns
predictorNames = string(features.Properties.VariableNames);
predictorNames = predictorNames(~ismember(predictorNames, ["User","Day","WindowID"]));

X_tbl = features(:, predictorNames);   % keep as table if needed
X = table2array(X_tbl);               % N x F
Y = features.User;                    % N x 1 (assumed 1..K)

% Basic checks
if any(isnan(X(:)))
    warning('X contains NaNs - replacing NaNs with column median.');
    % Replace NaN by column median
    for c = 1:size(X,2)
        col = X(:,c);
        col(isnan(col)) = median(col(~isnan(col)));
        X(:,c) = col;
    end
end

% -------------------------
% 3) Z-score (robust to zero sigma)
% -------------------------
mu = mean(X,1);
sigma = std(X,[],1);
sigma_fixed = sigma;
sigma_fixed(sigma_fixed == 0) = 1;   % avoid divide-by-zero for constant features
X = (X - mu) ./ sigma_fixed;

% -------------------------
% 4) Stratified 80/20 split (recommended)
% -------------------------
% Use cvpartition for stratified split
rng(0);
cv = cvpartition(Y, 'HoldOut', 0.20);
trainMask = training(cv);
testMask  = test(cv);

Xtrain_raw = X(trainMask, :);   % Ntrain x F
Ytrain_raw = Y(trainMask);
Xtest_raw  = X(testMask, :);    % Ntest x F
Ytest = Y(testMask);

% Transpose for neural net toolbox: features x samples
Xtrain = Xtrain_raw';
Xtest  = Xtest_raw';

% One-hot targets (neural net toolbox likes target vectors)
numClasses = numel(unique(Y));
Ttrain = full(ind2vec(Ytrain_raw'));   % numClasses x Ntrain
Ttest  = full(ind2vec(Ytest'));        % numClasses x Ntest

% -------------------------
% 5) Build & tune MLP (small sweep)
% -------------------------
hiddenCandidates = [20, 50, 100];   % try these
bestValAcc = -inf;
bestNet = [];
bestHidden = NaN;
useInternalValidation = true;  % set false to use all training data for training

for h = hiddenCandidates
    rng(0); % keep initialization same for comparability
    net = patternnet(h, 'trainscg');  % or try 'trainlm' for small data
    % If you want to use external test only, set internal ratios to zero:
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
    net.trainParam.max_fail = 12;   % early stopping patience
    net.performFcn = 'crossentropy';

    % Train
    [netTrained, tr] = train(net, Xtrain, Ttrain);

    % Evaluate on internal validation set if used (or on a portion of training)
    % Extract validation indices from 'tr' if available
    if useInternalValidation && isfield(tr, 'vind') && ~isempty(tr.vind)
        valIdx = tr.vind;   % indices w.r.t training samples
        Xval = Xtrain(:, valIdx);
        Tval = Ttrain(:, valIdx);
        YvalOut = netTrained(Xval);
        [~, yvalpred] = max(YvalOut, [], 1);
        yvalpred = yvalpred';
        YvalTrue = vec2ind(Tval)'; % convert back to class indices
        valAcc = mean(yvalpred == YvalTrue) * 100;
    else
        % fallback: evaluate on all training samples
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

% -------------------------
% 6) Final prediction on external test set
% -------------------------
Yhat_proba = net(Xtest);                   % numClasses x Ntest
[~, ypred] = max(Yhat_proba, [], 1);       % 1 x Ntest
ypred = ypred';                            % Ntest x 1

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

% Top-3 accuracy (optional)
[~, sortedIdx] = sort(Yhat_proba, 1, 'descend'); % numClasses x Ntest
top3 = sortedIdx(1:min(3,size(sortedIdx,1)), :);  % up to 3
top3acc = mean(any(top3' == Ytest, 2)) * 100;
fprintf('Top-3 accuracy: %.2f%%\n', top3acc);

% -------------------------
% 7) Plot confusion matrix (visual)
% -------------------------
figure;
plotconfusion(Ttest, Yhat_proba);
title(sprintf('MLP Confusion Matrix (hidden=%d) — Acc %.2f%%', hiddenUnits, acc));
saveas(gcf, 'MLP_confusion_randomsplit.png');

% -------------------------
% 8) Save model, scalers and metadata
% -------------------------
classLabels = unique(Y);  % assumptions
save("mlp_randomsplit.mat", "net", "mu", "sigma_fixed", "predictorNames", ...
    "trainMask", "testMask", "Ytest", "Xtest", "Ttest", "hiddenUnits", "bestValAcc", "classLabels");

fprintf('Model and metadata saved to mlp_randomsplit.mat\n');
