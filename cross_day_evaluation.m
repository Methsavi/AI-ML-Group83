% Train on Day1, test on Day2
clear; clc; close all;
rng(0);

forcedHiddenUnits = [];  

useInternalValidation = true; 
trainAlg = 'trainscg';
maxEpochs = 300;
maxFail = 12;

% Load data
if ~exist("features_dataset.mat", "file")
    error('features_dataset.mat not found in current folder. Put it in the project root.');
end
load("features_dataset.mat");   

if ~ismember("Day", features.Properties.VariableNames)
    error('features table must contain a ''Day'' column.');
end

% Prepare predictors & labels

predictorNames = string(features.Properties.VariableNames);
predictorNames = predictorNames(~ismember(predictorNames, ["User","Day","WindowID"]));
X_all = table2array(features(:, predictorNames));  
Y_all = features.User;                             
Days_all = features.Day;                           

% ensure Days_all is numeric or categorical, convert categorical/strings to numeric indices
if iscellstr(Days_all) || isstring(Days_all) || iscategorical(Days_all)
    [uniqueDays, ~, dayIdx] = unique(string(Days_all),'stable');
    Days_all = dayIdx;
else
    uniqueDays = unique(Days_all);
end

if numel(uniqueDays) < 2
    error('Need at least two different days for cross-day evaluation.');
end

% pick the first two distinct days
day1_value = uniqueDays(1);
day2_value = uniqueDays(2);

% create logical masks
if isstring(uniqueDays) || iscellstr(uniqueDays) || iscategorical(uniqueDays)
    trainMask = (Days_all == 1); 
    trainMask = (dayIdx == 1);
    testMask  = (dayIdx == 2);
else
    trainMask = (Days_all == day1_value);
    testMask  = (Days_all == day2_value);
end

% Extract sets
Xtrain_raw = X_all(trainMask, :);
Ytrain_raw = Y_all(trainMask);
Xtest_raw  = X_all(testMask, :);
Ytest = Y_all(testMask);

fprintf('Day1 samples (train): %d   Day2 samples (test): %d\n', size(Xtrain_raw,1), size(Xtest_raw,1));

%  Clean NaNs if any

for c = 1:size(Xtrain_raw,2)
    col = Xtrain_raw(:,c);
    if any(isnan(col))
        col(isnan(col)) = median(col(~isnan(col)));
        Xtrain_raw(:,c) = col;
    end
end
for c = 1:size(Xtest_raw,2)
    col = Xtest_raw(:,c);
    if any(isnan(col))
        col(isnan(col)) = median(col(~isnan(col)));
        Xtest_raw(:,c) = col;
    end
end

% Normalize using train stats 
mu = mean(Xtrain_raw,1);
sigma = std(Xtrain_raw,[],1);
sigma(sigma==0)=1;
Xtrain = (Xtrain_raw - mu) ./ sigma;
Xtest  = (Xtest_raw - mu) ./ sigma;  

Xtrain_t = Xtrain';
Xtest_t  = Xtest';

allClasses = unique(Y_all);
numClasses = numel(allClasses);

% ensure labels are index-1..K or map them to 1..K
[~, ~, Ytrain_idx] = unique(Ytrain_raw, 'stable');
[~, ~, Ytest_idx]  = unique(Ytest, 'stable');
[~, trainIdxMapped] = ismember(Ytrain_raw, allClasses);
[~, testIdxMapped]  = ismember(Ytest, allClasses);

Ttrain = full(ind2vec(trainIdxMapped'));
Ttest  = full(ind2vec(testIdxMapped'));

% Choose hidden units
hiddenUnits = [];
if ~isempty(forcedHiddenUnits)
    hiddenUnits = forcedHiddenUnits;
elseif exist("bestHidden","var") && ~isempty(bestHidden)
    hiddenUnits = bestHidden;
elseif exist("hiddenUnits","var") && ~isempty(hiddenUnits)
    
else
    hiddenUnits = 100; 
end

fprintf('Training MLP on Day1 with hidden units = %s\n', mat2str(hiddenUnits));


% Build & train network

net = patternnet(hiddenUnits, trainAlg);
if useInternalValidation
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio   = 15/100;
    net.divideParam.testRatio  = 15/100;
else
    net.divideParam.trainRatio = 1.0;
    net.divideParam.valRatio = 0.0;
    net.divideParam.testRatio = 0.0;
end
net.trainParam.epochs = maxEpochs;
net.trainParam.max_fail = maxFail;
net.performFcn = 'crossentropy';

[netTrained, tr] = train(net, Xtrain_t, Ttrain);


% Evaluate on Day2 
Yhat_proba = netTrained(Xtest_t);       
Yhat_proba = Yhat_proba';               
[~, ypred] = max(Yhat_proba, [], 2);   
ypred_labels = allClasses(ypred);

acc = mean(ypred_labels == Ytest) * 100;
fprintf('Cross-day accuracy (train Day1 -> test Day2): %.2f%%\n', acc);

% Confusion
C = confusionmat(Ytest, ypred_labels);
disp('Confusion matrix (rows=true, cols=predicted):');
disp(C);

% FAR / FRR / EER 

genuineScores = zeros(numel(Ytest),1);
impostorScores = [];

for i = 1:numel(Ytest)
    trueIdx = find(allClasses == Ytest(i));
    genuineScores(i) = Yhat_proba(i, trueIdx);
    otherIdx = setdiff(1:numClasses, trueIdx);
    impostorScores = [impostorScores; Yhat_proba(i, otherIdx)'];
end

thresholds = linspace(0,1,1000);
FAR = zeros(size(thresholds));
FRR = zeros(size(thresholds));
for t = 1:length(thresholds)
    thr = thresholds(t);
    FAR(t) = mean(impostorScores >= thr);
    FRR(t) = mean(genuineScores < thr);
end
[~, idxEER] = min(abs(FAR - FRR));
EER = mean([FAR(idxEER), FRR(idxEER)]) * 100;
EER_threshold = thresholds(idxEER);
fprintf('Cross-day EER = %.3f%% at threshold = %.3f\n', EER, EER_threshold);

% Plot FAR/FRR
figure;
plot(thresholds, FAR*100, 'r', 'LineWidth',1.5); hold on;
plot(thresholds, FRR*100, 'b', 'LineWidth',1.5);
plot(EER_threshold, EER, 'ko','MarkerFaceColor','k');
xlabel('Threshold'); ylabel('Error Rate (%)');
title(sprintf('Cross-day FAR/FRR — EER = %.3f%%', EER));
legend('FAR','FRR','EER','Location','best');
grid on;
saveas(gcf, 'MLP_CrossDay_FAR_FRR.png');

% Save results

mlp_crossday.net = netTrained;
mlp_crossday.mu = mu;
mlp_crossday.sigma = sigma;
mlp_crossday.hiddenUnits = hiddenUnits;
mlp_crossday.trainDay = day1_value;
mlp_crossday.testDay = day2_value;
mlp_crossday.accuracy = acc;
mlp_crossday.confusion = C;
mlp_crossday.EER = EER;
mlp_crossday.EER_threshold = EER_threshold;
mlp_crossday.Ytest = Ytest;
mlp_crossday.Yhat_proba = Yhat_proba;
mlp_crossday.ypred_labels = ypred_labels;

save('mlp_crossday.mat', 'mlp_crossday');

fprintf('Saved cross-day results to mlp_crossday.mat\n');
