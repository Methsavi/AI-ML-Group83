%% IMPORT ALL USER DATA (FD + MD)

clear; clc;

folder = "Dataset"; % folder name in your MATLAB project

files = dir(fullfile(folder, "*.csv"));

allData = table();

for i = 1:length(files)

    fileName = files(i).name;
    filePath = fullfile(files(i).folder, fileName);

    % Read CSV
    T = readtable(filePath);

    % ---------- Extract User ID ----------
    % Format = U<number>NW_<FD/MD>.csv
    tokens = regexp(fileName, 'U(\d+)NW_(FD|MD)\.csv', 'tokens');
    userID = str2double(tokens{1}{1});
    dayType = tokens{1}{2}; % FD or MD

    % Add columns
    T.User = repmat(userID, height(T), 1);
    T.Day  = repmat(string(dayType), height(T), 1);

    % Combine
    allData = [allData; T];
end

% Save final dataset
save("combined_dataset.mat", "allData");
disp("Combined data saved to combined_dataset.mat");
