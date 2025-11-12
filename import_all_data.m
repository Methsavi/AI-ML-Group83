clear; clc;

folder = "Dataset";

files = dir(fullfile(folder, "*.csv"));

allData = table();

for i = 1:length(files)

    fileName = files(i).name;
    filePath = fullfile(files(i).folder, fileName);

    % Read CSV
    T = readtable(filePath);

    %Extract User ID
    tokens = regexp(fileName, 'U(\d+)NW_(FD|MD)\.csv', 'tokens');
    userID = str2double(tokens{1}{1});
    dayType = tokens{1}{2}; 

    % Add columns
    T.User = repmat(userID, height(T), 1);
    T.Day  = repmat(string(dayType), height(T), 1);

    % Combine
    allData = [allData; T];
end

save("combined_dataset.mat", "allData");
disp("Combined data saved to combined_dataset.mat");

writetable(allData, "combined_dataset.csv");
disp("Combined data also saved to combined_dataset.csv");
