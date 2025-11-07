%% evaluate_auth_metrics.m
% Step 03 - Compute Authentication Metrics (FAR, FRR, EER) for trained MLP
clear; clc; close all;

% ----------------------------
% 1) Load trained MLP model
% ----------------------------
load("mlp_randomsplit.mat");   % loads: net, Xtest, Ytest, classLabels, etc.
fprintf('Loaded trained MLP model with %d hidden units.\n', hiddenUnits);

% ----------------------------
% 2) Get model prediction scores
% ----------------------------
Yhat_proba = net(Xtest);              % numClasses x Ntest
Yhat_proba = Yhat_proba';             % Ntest x numClasses
numUsers = numel(classLabels);

% ----------------------------
% 3) Build genuine & impostor scores
% ----------------------------
% For each test sample, take the model's probability for the true user as
% the genuine score, and probabilities for all *other* users as impostor scores.
genuineScores = zeros(numel(Ytest),1);
impostorScores = [];

for i = 1:numel(Ytest)
    trueUser = Ytest(i);
    genuineScores(i) = Yhat_proba(i, trueUser);

    % other users' probabilities are impostors
    otherIdx = setdiff(1:numUsers, trueUser);
    impostorScores = [impostorScores; Yhat_proba(i, otherIdx)'];
end

% ----------------------------
% 4) Compute FAR / FRR at various thresholds
% ----------------------------
thresholds = linspace(0,1,1000);
FAR = zeros(size(thresholds));
FRR = zeros(size(thresholds));

for t = 1:length(thresholds)
    thr = thresholds(t);

    % FAR: impostor accepted as genuine
    FAR(t) = mean(impostorScores >= thr);

    % FRR: genuine rejected
    FRR(t) = mean(genuineScores < thr);
end

% ----------------------------
% 5) Find EER (Equal Error Rate)
% ----------------------------
[~, idx] = min(abs(FAR - FRR));
EER = mean([FAR(idx), FRR(idx)]) * 100;
fprintf('Equal Error Rate (EER): %.2f%% at threshold=%.3f\n', EER, thresholds(idx));

% ----------------------------
% 6) Plot FAR and FRR curves
% ----------------------------
figure;
plot(thresholds, FAR*100, 'r', 'LineWidth',1.5); hold on;
plot(thresholds, FRR*100, 'b', 'LineWidth',1.5);
plot(thresholds(idx), EER, 'ko', 'MarkerFaceColor','k');
xlabel('Threshold'); ylabel('Error Rate (%)');
title(sprintf('FAR / FRR Curve — EER = %.2f%%', EER));
legend('FAR','FRR','EER point','Location','best');
grid on;
saveas(gcf, 'MLP_FAR_FRR_EER.png');

% ----------------------------
% 7) ROC Curve (optional)
% ----------------------------
figure;
[TPR, FPR, ~, AUC] = perfcurve(Ytest, Yhat_proba(:,1), 1); % example for class 1
plot(FPR, TPR, 'k', 'LineWidth',1.5);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('ROC Curve (example class) — AUC = %.3f', AUC));
grid on;
saveas(gcf, 'MLP_ROC_example.png');

fprintf('FAR/FRR/EER evaluation completed and saved.\n');