clear; clc;

% 1. Load combined dataset
load("combined_dataset.mat");   
fprintf("Loaded combined dataset.\n");

% 2. Remove mising values
before = height(allData);
allData = rmmissing(allData);
after = height(allData);
fprintf("Removed %d missing rows.\n", before - after);

% 3. Remove NaN + Inf values
allData = allData(~any(ismissing(allData), 2), :);   
numericCols = varfun(@isnumeric, allData, 'OutputFormat', 'uniform');
rowsOK = all(~isinf(allData{:, numericCols}), 2);    
allData = allData(rowsOK, :);
fprintf("Cleaned NaN/Inf values.\n");

try
    allData.Properties.VariableNames(1:7) = {'Ax','Ay','Az','Gx','Gy','Gz','Mag'};
    fprintf("Renamed Var1-7 to Ax, Ay, Az, Gx, Gy, Gz, Mag.\n");
catch ME
    fprintf("Warning: Could not rename columns. Check if they are already named correctly.\n");
    disp(ME.message);
end

% 4. Normalize Sensor Columns
signals = ["Ax","Ay","Az","Gx","Gy","Gz"];
fprintf("Normalizing sensor columns (z-score)...\n");

for s = signals
    if ismember(s, allData.Properties.VariableNames)
        allData.(s) = normalize(allData.(s));  
        fprintf(" - Normalized %s.\n", s);
    else
        fprintf(" - Warning: %s not found. Skipping normalization.\n", s);
    end
end
fprintf("Normalization complete.\n");

% 5. Segment into 3-second Windows
fs = 30;                  
windowSize = 3 * fs;      
stepSize   = windowSize;  

users = unique(allData.User);
processedData = [];
fprintf("Segmenting data into windows...\n");
windowID = 1;

for u = users'
    userData = allData(allData.User == u, :);
    N = height(userData);
    
    for i = 1:stepSize:(N - windowSize + 1)
        segment = userData(i:i + windowSize - 1, :);
       
        if all(segment.Day == segment.Day(1))
            seg.WindowID = windowID;
            seg.User = u;
            seg.Day = segment.Day(1);
            seg.Data = {segment}; 
            processedData = [processedData; struct2table(seg)];
            windowID = windowID + 1;
        end
    end
end

fprintf("Segmented into %d windows.\n", height(processedData));

save("preprocessed_dataset.mat", "processedData");
fprintf("Preprocessing completed. File saved: preprocessed_dataset.mat \n");

writetable(processedData, "preprocessed_dataset.csv");
fprintf("Saved CSV version: preprocessed_dataset.csv \n");