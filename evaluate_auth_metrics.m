% Compute Authentication Metrics (FAR, FRR, EER) for trained MLP
clear; clc; close all;

load("mlp_randomsplit.mat"); 
fprintf('Loaded trained MLP model with %d hidden units.\n', hiddenUnits);

% Get model prediction scores
Yhat_proba = net(Xtest);              
Yhat_proba = Yhat_proba';             
numUsers = numel(classLabels);

genuineScores = zeros(numel(Ytest),1);
impostorScores = [];

for i = 1:numel(Ytest)
    trueUser = Ytest(i);
    genuineScores(i) = Yhat_proba(i, trueUser);
    otherIdx = setdiff(1:numUsers, trueUser);
    impostorScores = [impostorScores; Yhat_proba(i, otherIdx)'];
end

% Compute FAR / FRR at various thresholds
thresholds = linspace(0,1,1000);
FAR = zeros(size(thresholds));
FRR = zeros(size(thresholds));

for t = 1:length(thresholds)
    thr = thresholds(t);

    FAR(t) = mean(impostorScores >= thr);

    FRR(t) = mean(genuineScores < thr);
end

% Find EER 
[~, idx] = min(abs(FAR - FRR));
EER = mean([FAR(idx), FRR(idx)]) * 100;
fprintf('Equal Error Rate (EER): %.2f%% at threshold=%.3f\n', EER, thresholds(idx));

% Plot FAR and FRR curves
figure;
plot(thresholds, FAR*100, 'r', 'LineWidth',1.5); hold on;
plot(thresholds, FRR*100, 'b', 'LineWidth',1.5);
plot(thresholds(idx), EER, 'ko', 'MarkerFaceColor','k');
xlabel('Threshold'); ylabel('Error Rate (%)');
title(sprintf('FAR / FRR Curve — EER = %.2f%%', EER));
legend('FAR','FRR','EER point','Location','best');
grid on;
saveas(gcf, 'MLP_FAR_FRR_EER.png');

% ROC Curve
figure;
[TPR, FPR, ~, AUC] = perfcurve(Ytest, Yhat_proba(:,1), 1);
plot(FPR, TPR, 'k', 'LineWidth',1.5);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('ROC Curve (example class) — AUC = %.3f', AUC));
grid on;
saveas(gcf, 'MLP_ROC_example.png');

fprintf('FAR/FRR/EER evaluation completed and saved.\n');