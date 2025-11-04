%% STEP 9: MODEL EVALUATION & VISUALIZATION
% -------------------------------------------------------------
% Loads:
%   - trained_models.mat (SVM, Random Forest, KNN)
%   - features_dataset.mat (features)
%
% Outputs:
%   - Confusion matrices
%   - Classification accuracy
%   - Per-class performance
%   - Model comparison table
% -------------------------------------------------------------

clear; clc;

%% Load models and dataset
load("trained_models.mat");
load("features_dataset.mat");

%% Extract predictors and labels
predictorNames = string(features.Properties.VariableNames);
predictorNames = predictorNames(~ismember(predictorNames, ["User", "Day", "WindowID"]));

X = features{:, predictorNames};
Y = features.User;

%% Manual Train-Test Split (same as Step 8)
n = length(Y);
idx = randperm(n);

testCount = round(0.2 * n);

testIdx  = idx(1:testCount);
trainIdx = idx(testCount+1:end);

Xtrain = X(trainIdx, :);
Ytrain = Y(trainIdx);

Xtest  = X(testIdx, :);
Ytest  = Y(testIdx);

%% Evaluate each model
% ----- SVM -----
pred_svm = predict(model_svm, Xtest);
acc_svm = mean(pred_svm == Ytest);

% ----- Random Forest -----
[pred_rf, ~] = predict(model_rf, Xtest);
pred_rf = str2double(pred_rf);  % convert labels
acc_rf = mean(pred_rf == Ytest);

% ----- KNN -----
pred_knn = predict(model_knn, Xtest);
acc_knn = mean(pred_knn == Ytest);

%% Display accuracy summary
fprintf("\n--- MODEL ACCURACY ---\n");
fprintf("SVM Accuracy: %.2f%%\n", acc_svm * 100);
fprintf("Random Forest Accuracy: %.2f%%\n", acc_rf * 100);
fprintf("KNN Accuracy: %.2f%%\n\n", acc_knn * 100);

%% Confusion Matrices
figure;
confusionchart(Ytest, pred_rf);
title("Random Forest Confusion Matrix");

figure;
confusionchart(Ytest, pred_svm);
title("SVM Confusion Matrix");

figure;
confusionchart(Ytest, pred_knn);
title("KNN Confusion Matrix");

%% Classification Report (per-class accuracy)
users = unique(Y);

fprintf("--- PER-CLASS PERFORMANCE ---\n");
for u = users'
    idxU = (Ytest == u);
    accU = mean(pred_rf(idxU) == u);   % using best model RF
    fprintf("User %d Accuracy: %.2f%%\n", u, accU * 100);
end

%% Save visuals as images
saveas(gcf, "confusion_matrix_best_model.png");

fprintf("\nEvaluation completed.\n");
