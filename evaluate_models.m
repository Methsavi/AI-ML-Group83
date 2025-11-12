% Model evaluation and visualization

clear; clc;

load("trained_models.mat");
load("features_dataset.mat");

% Extract predictors and labels
predictorNames = string(features.Properties.VariableNames);
predictorNames = predictorNames(~ismember(predictorNames, ["User", "Day", "WindowID"]));

X = features{:, predictorNames};
Y = features.User;

% Manual Train-Test Split
n = length(Y);
idx = randperm(n);

testCount = round(0.2 * n);

testIdx  = idx(1:testCount);
trainIdx = idx(testCount+1:end);

Xtrain = X(trainIdx, :);
Ytrain = Y(trainIdx);

Xtest  = X(testIdx, :);
Ytest  = Y(testIdx);

%  Evaluate each model
% SVM
pred_svm = predict(model_svm, Xtest);
acc_svm = mean(pred_svm == Ytest);

% Random Forest
[pred_rf, ~] = predict(model_rf, Xtest);
pred_rf = str2double(pred_rf); 
acc_rf = mean(pred_rf == Ytest);

% KNN
pred_knn = predict(model_knn, Xtest);
acc_knn = mean(pred_knn == Ytest);

fprintf("Model accurasy \n \n");
fprintf("SVM Accuracy is %.2f%%\n", acc_svm * 100);
fprintf("Randon Forest Accuracy is %.2f%%\n", acc_rf * 100);
fprintf("KNN Accuracy is %.2f%%\n\n", acc_knn * 100);

% Confusion Matrics
figure;
confusionchart(Ytest, pred_rf);
title("Random Forest Confusion Matrix");

figure;
confusionchart(Ytest, pred_svm);
title("SVM Confusiob Matrix");

figure;
confusionchart(Ytest, pred_knn);
title("KNN Confusion Matrix");

% Classificasion Report (per-class accuracy)
users = unique(Y);

fprintf("Performance of each class \n \n");
for u = users'
    idxU = (Ytest == u);
    accU = mean(pred_rf(idxU) == u);
    fprintf("User %d Accuracy is %.2f%%\n", u, accU * 100);
end

saveas(gcf, "confusion_matrix_best_model.png");
fprintf("\nEvaluation completed.\n");
