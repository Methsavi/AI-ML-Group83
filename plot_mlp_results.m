% === Load trained MLP model results ===
load('mlp_randomsplit.mat');

% === Generate model predictions and probabilities ===
Yhat_proba = net(Xtest);              % Predicted probabilities (numClasses x numSamples)
[~, ypred] = max(Yhat_proba, [], 1);  % Predicted class indices

% === Plot Confusion Matrix ===
figure;
confusionchart(Ytest, ypred);
title('Confusion Matrix - MLP Random Split');

% === Compute ROC curve, AUC, FAR, FRR, and EER (one-vs-all per class) ===
numClasses = size(Yhat_proba, 1);
AUC = zeros(numClasses, 1);
EER = zeros(numClasses, 1);
EER_threshold = zeros(numClasses, 1);

for c = 1:numClasses
    % Binary ground truth for class c
    targets_c = (Ytest == c);
    % Predicted probabilities for class c
    scores_c = Yhat_proba(c, :)';
    
    % Compute ROC and AUC for this class
    [~, ~, ~, AUC(c)] = perfcurve(targets_c, scores_c, 1);
    
    % Compute FAR and FRR for thresholds
    thresholds = linspace(0, 1, 100);
    FAR = zeros(size(thresholds));
    FRR = zeros(size(thresholds));

    for i = 1:length(thresholds)
        threshold = thresholds(i);
        predictions = scores_c > threshold;
        FAR(i) = sum((predictions == 1) & (targets_c == 0)) / sum(targets_c == 0);
        FRR(i) = sum((predictions == 0) & (targets_c == 1)) / sum(targets_c == 1);
    end

    % Find Equal Error Rate (EER)
    [~, idxEER] = min(abs(FAR - FRR));
    EER(c) = mean([FAR(idxEER), FRR(idxEER)]);
    EER_threshold(c) = thresholds(idxEER);
end

% === Display AUC and EER results in a table ===
disp('Performance Summary (AUC and EER per Class):');
fprintf('-----------------------------------------------\n');
fprintf(' Class |    AUC     |   EER     | Threshold\n');
fprintf('-----------------------------------------------\n');
for c = 1:numClasses
    fprintf('  %3d   |  %.4f  |  %.4f  |  %.3f\n', c, AUC(c), EER(c), EER_threshold(c));
end
fprintf('-----------------------------------------------\n');
fprintf('Mean AUC: %.4f\n', mean(AUC));
fprintf('Mean EER: %.4f\n', mean(EER));

% === Plot AUC per class (Bar chart) ===
figure;
bar(AUC);
xlabel('Class Index');
ylabel('AUC Value');
title('AUC per Class - MLP Random Split');
grid on;
ylim([0.95 1.01]);
text(1:numClasses, AUC, string(round(AUC, 4)), 'HorizontalAlignment','center', ...
     'VerticalAlignment','bottom', 'FontSize',8);

% === Plot EER per class (Bar chart) ===
figure;
bar(EER);
xlabel('Class Index');
ylabel('EER (Equal Error Rate)');
title('EER per Class - MLP Random Split');
grid on;
ylim([0 max(EER)*1.2]);
text(1:numClasses, EER, string(round(EER, 4)), 'HorizontalAlignment','center', ...
     'VerticalAlignment','bottom', 'FontSize',8);

% === Plot FAR and FRR for one example class ===
exampleClass = 1; % Change to visualize a specific class
targets = (Ytest == exampleClass);
outputs = Yhat_proba(exampleClass, :)';

thresholds = linspace(0, 1, 100);
FAR = zeros(size(thresholds));
FRR = zeros(size(thresholds));

for i = 1:length(thresholds)
    threshold = thresholds(i);
    predictions = outputs > threshold;
    FAR(i) = sum((predictions == 1) & (targets == 0)) / sum(targets == 0);
    FRR(i) = sum((predictions == 0) & (targets == 1)) / sum(targets == 1);
end

[~, idxEER] = min(abs(FAR - FRR));
EER_value = mean([FAR(idxEER), FRR(idxEER)]);

figure;
plot(thresholds, FAR, 'r', 'LineWidth', 1.5); hold on;
plot(thresholds, FRR, 'b', 'LineWidth', 1.5);
plot(thresholds(idxEER), FAR(idxEER), 'ko', 'MarkerFaceColor', 'k');
text(thresholds(idxEER), FAR(idxEER), ...
    sprintf('  EER = %.4f', EER_value), ...
    'VerticalAlignment', 'bottom');
xlabel('Threshold');
ylabel('Error Rate');
legend('FAR', 'FRR', 'EER Point');
title(sprintf('FAR vs FRR (Class %d)', exampleClass));
grid on;

fprintf('\nExample Class %d → EER = %.4f at threshold = %.3f\n', ...
    exampleClass, EER_value, thresholds(idxEER));
