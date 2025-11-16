% Leave-One-User-Out (Cross-User) Evaluation

clear; clc; close all;
rng(0);

% Configurations
hiddenUnits = 100;
trainAlg = 'trainscg';
maxEpochs = 300;
maxFail = 12;

if ~exist('features_dataset.mat', 'file')
    error('features_dataset.mat not found in current folder.');
end

load('features_dataset.mat');  

if ~ismember("User", features.Properties.VariableNames)
    error('features table must contain a "User" column.');
end

fprintf('Loaded dataset with %d samples and %d features.\n', size(features,1), size(features,2)-1);

predictorNames = string(features.Properties.VariableNames);
predictorNames = predictorNames(~ismember(predictorNames, ["User","Day","WindowID"]));

X_all = table2array(features(:, predictorNames));
Y_all = features.User;

allUsers = unique(Y_all);
numUsers = numel(allUsers);

fprintf('Found %d unique users.\n', numUsers);

acc_per_user = zeros(numUsers,1);
eer_per_user = zeros(numUsers,1);

% Cross-User Loop
for u = 1:numUsers
    testUser = allUsers(u);
    trainMask = (Y_all ~= testUser);
    testMask  = (Y_all == testUser);

    Xtrain_raw = X_all(trainMask,:);
    Ytrain_raw = Y_all(trainMask);
    Xtest_raw  = X_all(testMask,:);
    Ytest      = Y_all(testMask);

    % Normalize using train stats
    mu = mean(Xtrain_raw,1);
    sigma = std(Xtrain_raw,[],1);
    sigma(sigma==0) = 1;
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest  = (Xtest_raw - mu) ./ sigma;

    Xtrain_t = Xtrain';
    Xtest_t  = Xtest';

    % Encode labels based only on training users
    trainClasses = unique(Ytrain_raw);
    numTrainClasses = numel(trainClasses);
    [~, trainIdx] = ismember(Ytrain_raw, trainClasses);
    [~, testIdx]  = ismember(Ytest, trainClasses);
    Ttrain = full(ind2vec(trainIdx'));

    
    % Train Model
    net = patternnet(hiddenUnits, trainAlg);
    net.trainParam.epochs = maxEpochs;
    net.trainParam.max_fail = maxFail;
    net.performFcn = 'crossentropy';
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio   = 15/100;
    net.divideParam.testRatio  = 15/100;

    [netTrained, ~] = train(net, Xtrain_t, Ttrain);

    % Predict on test user
    
    Yhat_proba = netTrained(Xtest_t);
    Yhat_proba = Yhat_proba';
    [~, ypred] = max(Yhat_proba, [], 2);
    ypred_labels = trainClasses(ypred);

    % Accuracy 
    acc = mean(ypred_labels == Ytest) * 100;
    acc_per_user(u) = acc;

    % FAR / FRR / EER
    if ismember(testUser, trainClasses)
        genuineScores = zeros(numel(Ytest),1);
        impostorScores = [];
        for i = 1:numel(Ytest)
            trueIdx = find(trainClasses == Ytest(i));
            genuineScores(i) = Yhat_proba(i, trueIdx);
            otherIdx = setdiff(1:numTrainClasses, trueIdx);
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
        eer_per_user(u) = EER;
    else
        eer_per_user(u) = NaN;
    end

    fprintf('User %s -> Acc = %.2f%% | EER = %.2f%%\n', string(testUser), acc, eer_per_user(u));
end

validEER = eer_per_user(~isnan(eer_per_user));
fprintf('\n Cross-User Summary \n');
fprintf('Mean Accuracy (%%) is %.2f\n', mean(acc_per_user));
fprintf('Mean EER (%%) is %.2f\n', mean(validEER));

save('mlp_crossuser.mat', 'acc_per_user', 'eer_per_user', 'allUsers');

figure;
bar(acc_per_user);
xlabel('User Index');
ylabel('Accuracy (%)');
title('Cross-User Accuracy per User');
grid on;
saveas(gcf, 'MLP_CrossUser_Accuracy.png');

figure;
bar(eer_per_user);
xlabel('User Index');
ylabel('EER (%)');
title('Cross-User EER per User');
grid on;
saveas(gcf, 'MLP_CrossUser_EER.png');

fprintf('Saved results to mlp_crossuser.mat\n');
