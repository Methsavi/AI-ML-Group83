%FEATURE EXTRACTION 

clear; clc;

load("preprocessed_dataset.mat");
signals = ["Ax","Ay","Az","Gx","Gy","Gz"];
features = []; 

fprintf("Starting feature extraction for %d windows...\n", height(processedData));

% Process each window
for i = 1:height(processedData)
    
    feat = struct();
    
    window = processedData.Data{i};    
    feat.WindowID = processedData.WindowID(i);
    feat.User     = processedData.User(i);
    feat.Day      = processedData.Day(i);

    for s = signals
        x = window.(s);

        feat.(s + "_mean")  = mean(x);
        feat.(s + "_std")   = std(x);
        feat.(s + "_var")   = var(x);
        feat.(s + "_min")   = min(x);
        feat.(s + "_max")   = max(x);
        feat.(s + "_range") = max(x) - min(x);
        feat.(s + "_rms")   = rms(x);
        feat.(s + "_zcr")   = sum(diff(sign(x)) ~= 0);
        
        X = fft(x);
        P = abs(X).^2;                    
        feat.(s + "_energy")  = sum(P);
        P_norm = P / sum(P);               
        feat.(s + "_entropy") = -sum(P_norm .* log2(P_norm + eps));
        [~, idx] = max(P);
        feat.(s + "_domfreq") = idx;     
    end
    
    features = [features; struct2table(feat)];
end

fprintf("Feature extraction completed successfully!\n");

save("features_dataset.mat", "features");
writetable(features, "features_dataset.csv"); 
disp("File saved as features_dataset.mat and features_dataset.csv");