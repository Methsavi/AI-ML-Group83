% Compare MLP configurations 
clear; clc;

load('mlp_optimization_results.mat'); 

if isstruct(results)
    results = struct2table(results);
end

% Extract columns
hiddenLayers = results.HiddenLayers;
accuracies = results.Accuracy;

% Convert cell arrays like {'[100 50]'} to strings for labeling
for i = 1:numel(hiddenLayers)
    if iscell(hiddenLayers{i})
        hiddenLayers{i} = mat2str(cell2mat(hiddenLayers{i}));
    elseif isnumeric(hiddenLayers{i})
        hiddenLayers{i} = mat2str(hiddenLayers{i});
    elseif ~ischar(hiddenLayers{i})
        hiddenLayers{i} = string(hiddenLayers{i});
    end
end

% Combine duplicates: average accuracy for same config 
[uniqueConfigs, ~, idx] = unique(hiddenLayers);
meanAccuracies = accumarray(idx, accuracies, [], @mean);

% Plot bar chart
figure;
bar(categorical(uniqueConfigs), meanAccuracies);
xlabel('Hidden Layer Configuration');
ylabel('Accuracy (%)');
title('Comparison of MLP Configurations');
grid on;

% Annotate values on bars 
text(1:numel(uniqueConfigs), meanAccuracies, ...
    string(round(meanAccuracies, 2)), ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize',8);

fprintf('Comparison chart generated successfully.\n');
