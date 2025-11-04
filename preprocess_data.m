%% STEP 5: DATA PREPROCESSING (FINAL CLEAN VERSION)
% -------------------------------------------------------------
% 1. Load combined dataset
% 2. Remove missing + Inf values
% 3. Normalize (z-score)
% 4. Segment into 3-second windows
% -------------------------------------------------------------

clear; clc;

%%  1. Load combined dataset
load("combined_dataset.mat");   % loads allData

fprintf("Loaded combined dataset.\n");

%% 2. Remove missing values
before = height(allData);
allData = rmmissing(allData);
after = height(allData);
fprintf("Removed %d missing rows.\n", before - after);

%% 3. Remove NaN + Inf values
allData = allData(~any(ismissing(allData), 2), :);   % remove NaN rows

numericCols = varfun(@isnumeric, allData, 'OutputFormat', 'uniform');
rowsOK = all(~isinf(allData{:, numericCols}), 2);    % remove Inf rows
allData = allData(rowsOK, :);

fprintf("Cleaned NaN/Inf values.\n");

%% 4. Normalize Sensor Columns
signals = ["Ax","Ay","Az","Gx","Gy","Gz"];  % change names if needed

for s = signals
    if ismember(s, allData.Properties.VariableNames)
        allData.(s) = normalize(allData.(s));   % z-score normalization
    end
end

fprintf("Normalized all numeric signals.\n");

%% 5. Segment into 3-second Windows
fs = 30;                   % sampling rate
windowSize = 3 * fs;       % 3 seconds → 90 samples
stepSize   = windowSize;   % non-overlapping

users = unique(allData.User);
processedData = [];

fprintf("Segmenting data into windows...\n");

windowID = 1;

for u = users'
    userData = allData(allData.User == u, :);
    N = height(userData);

    for i = 1:stepSize:(N - windowSize)
        segment = userData(i:i + windowSize - 1, :);

        seg.WindowID = windowID;
        seg.User = u;
        seg.Day = segment.Day(1);
        seg.Data = {segment};

        processedData = [processedData; struct2table(seg)];

        windowID = windowID + 1;
    end
end

fprintf("Segmented into %d windows.\n", height(processedData));

%% Rename sensor columns to proper names
% Var1–Var7 → Ax Ay Az Gx Gy Gz Mag

for i = 1:height(processedData)
    processedData.Data{i}.Properties.VariableNames(1:7) = ...
        {'Ax','Ay','Az','Gx','Gy','Gz','Mag'};
end

save("preprocessed_dataset.mat","processedData");


%% Save processed dataset
save("preprocessed_dataset.mat", "processedData");
fprintf("Preprocessing completed... File saved: preprocessed_dataset.mat \n");
