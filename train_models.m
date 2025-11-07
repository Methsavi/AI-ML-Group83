%% STEP 8: MODEL TRAINING 
clear; clc;

load("features_dataset.mat");

% Extract predictor names
predictorNames = string(features.Properties.VariableNames);
predictorNames = predictorNames(~ismember(predictorNames, ["User", "Day", "WindowID"]));

X = features{:, predictorNames};
Y = features.User;

%% Manual Train-Test Split (20%)
n = length(Y);
idx = randperm(n);

testCount = round(0.2 * n);

testIdx  = idx(1:testCount);
trainIdx = idx(testCount+1:end);

Xtrain = X(trainIdx, :);
Ytrain = Y(trainIdx);

Xtest  = X(testIdx, :);
Ytest  = Y(testIdx);

fprintf("Training set size: %d samples\n", length(Ytrain));
fprintf("Testing  set size: %d samples\n", length(Ytest));

%% Train Models
model_svm = fitcecoc(Xtrain, Ytrain);
model_rf  = TreeBagger(50, Xtrain, Ytrain, 'Method', 'classification');
model_knn = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 5);

%% Evaluate
pred_svm = predict(model_svm, Xtest);

[pred_rf, ~] = predict(model_rf, Xtest);
pred_rf = str2double(pred_rf);  % Convert class labels

pred_knn = predict(model_knn, Xtest);

acc_svm = mean(pred_svm == Ytest);
acc_rf  = mean(pred_rf  == Ytest);
acc_knn = mean(pred_knn == Ytest);

fprintf("SVM Accuracy: %.2f%%\n", acc_svm * 100);
fprintf("Random Forest Accuracy: %.2f%%\n", acc_rf * 100);
fprintf("KNN Accuracy: %.2f%%\n", acc_knn * 100);

%% Save models
save("trained_models.mat", "model_svm", "model_rf", "model_knn");

fprintf("Models saved to trained_models.mat\n");

fprintf("checking my git commit");
