%% STEP 02 — Optimization Experiments for MLP
% Tests different hidden layers, neurons, and train/test ratios
% to find best-performing configuration.

clear; clc; rng(1);
load("features_dataset.mat");

% -------------------------
% 1) Data preparation
% -------------------------
predictorNames = string(features.Properties.VariableNames);
predictorNames = predictorNames(~ismember(predictorNames, ["User","Day","WindowID"]));

X = table2array(features(:, predictorNames));
Y = features.User;

% Replace NaNs if any
for c = 1:size(X,2)
    col = X(:,c);
    if any(isnan(col))
        col(isnan(col)) = median(col(~isnan(col)));
        X(:,c) = col;
    end
end

% Z-score normalize
mu = mean(X,1);
sigma = std(X,[],1);
sigma(sigma==0)=1;
X = (X - mu) ./ sigma;

numClasses = numel(unique(Y));

% -------------------------
% 2) Define experiment grid
% -------------------------
hiddenConfigs = {
    [20], [50], [100], ...
    [50 20], [100 50]
};
trainRatios = [0.6, 0.7, 0.8];
algorithm = 'trainscg';  % or try 'trainlm' for smaller sets

results = table();

expID = 1;
for h = 1:numel(hiddenConfigs)
    for trRatio = trainRatios

        % Stratified split
        cv = cvpartition(Y, 'HoldOut', 1-trRatio);
        Xtrain = X(training(cv), :)';
        Xtest  = X(test(cv), :)';
        Ytrain = Y(training(cv));
        Ytest  = Y(test(cv));

        Ttrain = full(ind2vec(Ytrain'));
        Ttest  = full(ind2vec(Ytest'));

        % -------------------------
        % 3) Build and train network
        % -------------------------
        net = patternnet(hiddenConfigs{h}, algorithm);
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio   = 15/100;
        net.divideParam.testRatio  = 15/100;
        net.trainParam.epochs = 300;
        net.trainParam.max_fail = 12;
        net.performFcn = 'crossentropy';

        [netTrained, tr] = train(net, Xtrain, Ttrain);

        % -------------------------
        % 4) Evaluate
        % --
        Yhat = netTrained(Xtest);
        [~, ypred] = max(Yhat, [], 1);
        ypred = ypred';
        acc = mean(ypred == Ytest) * 100;

        % Record results
        results.ExpID(expID,1) = expID;
        results.HiddenLayers{expID,1} = mat2str(hiddenConfigs{h});
        results.TrainRatio(expID,1) = trRatio;
        results.Accuracy(expID,1) = acc;
        results.BestEpoch(expID,1) = tr.best_epoch;
        expID = expID + 1;

        fprintf('Hidden=%s | TrainRatio=%.2f | Acc=%.2f%%\n', ...
            mat2str(hiddenConfigs{h}), trRatio, acc);
    end
end

% -------------------------
% 5) Show and save results
% -------------------------
[~, bestIdx] = max(results.Accuracy);
bestConfig = results(bestIdx, :);
disp('--- Summary of all runs ---');
disp(results);
disp('--- Best Configuration ---');
disp(bestConfig);

save("mlp_optimization_results.mat", "results", "bestConfig");