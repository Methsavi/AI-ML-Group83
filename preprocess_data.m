%% STEP 5: DATA PREPROCESSING (FINAL CLEAN VERSION - CORRECTED)
% -------------------------------------------------------------
% 1. Load combined dataset
% 2. Remove missing + Inf values
% 3. Rename columns (Var1 -> Ax, etc.)
% 4. Normalize (z-score)
% 5. Segment into 3-second windows
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

%% 3.5 RENAME COLUMNS (*** NEW/MOVED STEP ***)
% Rename columns BEFORE normalizing.
% Assuming Var1-Var7 are your 7 sensor columns in order.
try
    allData.Properties.VariableNames(1:7) = {'Ax','Ay','Az','Gx','Gy','Gz','Mag'};
    fprintf("Renamed Var1-7 to Ax, Ay, Az, Gx, Gy, Gz, Mag.\n");
catch ME
    fprintf("Warning: Could not rename columns. Check if they are already named correctly.\n");
    disp(ME.message);
end

%% 4. Normalize Sensor Columns
% NOTE: 'Mag' is not normalized here, only the 6 core signals.
% You can add "Mag" to this list if you want to normalize it too.
signals = ["Ax","Ay","Az","Gx","Gy","Gz"];
fprintf("Normalizing sensor columns (z-score)...\n");

for s = signals
    if ismember(s, allData.Properties.VariableNames)
        allData.(s) = normalize(allData.(s));   % z-score normalization
        fprintf(" - Normalized %s.\n", s);
    else
        fprintf(" - Warning: %s not found. Skipping normalization.\n", s);
    end
end
fprintf("Normalization complete.\n");

%% 5. Segment into 3-second Windows
fs = 30;                   % sampling rate (approx 30-32 Hz)
windowSize = 3 * fs;       % 3 seconds -> 90 samples
stepSize   = windowSize;   % non-overlapping

users = unique(allData.User);
processedData = [];
fprintf("Segmenting data into windows...\n");
windowID = 1;

for u = users'
    % Get all data for one user
    userData = allData(allData.User == u, :);
    N = height(userData);
    
    % Loop through the user's data in window-sized steps
    % We use (N - windowSize + 1) to ensure we capture the last possible full window
    for i = 1:stepSize:(N - windowSize + 1)
        segment = userData(i:i + windowSize - 1, :);
        
        % Safety Check: Ensure the window doesn't span two different days (FD/MD)
        if all(segment.Day == segment.Day(1))
            seg.WindowID = windowID;
            seg.User = u;
            seg.Day = segment.Day(1);
            
            % Store the data segment itself.
            % The columns are already named correctly (Ax, Ay, etc.)
            seg.Data = {segment}; 
            
            processedData = [processedData; struct2table(seg)];
            windowID = windowID + 1;
        end
    end
end

fprintf("Segmented into %d windows.\n", height(processedData));

%% 6. Save processed dataset
% The final renaming loop from your old code is no longer needed
% because the columns were already named correctly before segmentation.
save("preprocessed_dataset.mat", "processedData");
fprintf("Preprocessing completed. File saved: preprocessed_dataset.mat \n");

% ADD THIS LINE TO SAVE AS .CSV 
writetable(processedData, "preprocessed_dataset.csv");
fprintf("Saved CSV version: preprocessed_dataset.csv \n");