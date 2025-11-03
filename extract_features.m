%% STEP 7: FEATURE EXTRACTION
% -------------------------------------------------------------
% Input:  preprocessed_dataset.mat (processedData)
% Output: features_dataset.mat     (features table)
% -------------------------------------------------------------

clear; clc;

%% Load segmented dataset
load("preprocessed_dataset.mat");   % loads processedData

signals = ["Ax","Ay","Az","Gx","Gy","Gz"];
features = [];

%% Process each window
for i = 1:height(processedData)
    
    window = processedData.Data{i};     % 90-sample table
    feat.WindowID = processedData.WindowID(i);
    feat.User     = processedData.User(i);
    feat.Day      = processedData.Day(i);

    % Extract features for each signal
    for s = signals
        x = window.(s);

        % Time-domain features
        feat.(s + "_mean")  = mean(x);
        feat.(s + "_std")   = std(x);
        feat.(s + "_var")   = var(x);
        feat.(s + "_min")   = min(x);
        feat.(s + "_max")   = max(x);
        feat.(s + "_range") = max(x) - min(x);
        feat.(s + "_rms")   = rms(x);
        feat.(s + "_zcr")   = sum(diff(sign(x)) ~= 0);

        % Frequency-domain features
        X = fft(x);
        P = abs(X).^2;                     % Power spectrum

        feat.(s + "_energy")  = sum(P);
        P_norm = P / sum(P);               % Normalized power
        feat.(s + "_entropy") = -sum(P_norm .* log2(P_norm + eps));

        [~, idx] = max(P);
        feat.(s + "_domfreq") = idx;       % Dominant frequency index
    end

    features = [features; struct2table(feat)];
end
        x = window{:, s};  % Corrected to access the signal data as a column
    % Clear the temporary feature structure for the next window
    feat = struct();

%% Save extracted feature dataset
save("features_dataset.mat", "features");

disp("Feature extraction completed successfully! File saved as features_dataset.mat");
