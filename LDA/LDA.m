%%

clc, clear, close all

data = readtable("data/forest/training.csv");

labels = data.class; % this stays as cell/string/categorical
labels = categorical(labels); % convert labels from cell to category
data.class = []; % remove the non-numeric column
features = table2array(data); % convert remaining numeric part

% summary(labels) % Tells no undefined
% summary(features) % Tells no NA

%%


