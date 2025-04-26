
%% Load Data

clc; clear; close all;

data = readtable("data/forest/training.csv");

labels = categorical(data.class);   % All labels 198 by 1
classes = categories(labels);   % 4 classes
K = numel(classes);

data.class = [];        % Delete class column

X = table2array(data)'; % n x p = 27 by 198
Y = grp2idx(labels)';   % Index vector 1 by 198 (1 to K)

[n, p] = size(X);       % n = 27 / p = 198

%% Impurity Measures

% 1 - max(Pj)
function r = misclassification_error(Y)
    counts = histcounts(Y, 1:(max(Y)+1));
    probs = counts / sum(counts);

    r = 1 - max(probs);
end

% 1 - SUM[(Pj)^2]
function g = gini_index(Y)
    counts = histcounts(Y, 1:(max(Y)+1));
    probs = counts / sum(counts);

    g = 1 - sum(probs.^2);
end

%%

