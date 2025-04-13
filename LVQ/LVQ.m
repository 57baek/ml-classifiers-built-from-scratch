%%

clc; clear; close all;

data = readtable("data/forest/training.csv");
%data = readtable("data/forest/testing.csv");

labels = categorical(data.class);
classes = categories(labels);
K = numel(classes);
data.class = [];

X = table2array(data)';
y = grp2idx(labels)';  % numeric labels
[n, p] = size(X);

%%
L_pro = [3,6,8]; % different number of prototypes per class
N = 100 * p; % number of training steps (iterations)
alpha0 = 0.1; % initial learning rate
alphas = linspace(alpha0, 0.01, N); % decreasing learning rate

for L = L_pro % loop over each L in the set

    M = zeros(n, K * L); % all prototypes (like 1,1,1,2,2,2,3,3,3,4,4,4)
    pro_labels = zeros(1, K * L);  % labels of all prototypes
    
    % Initialization with random values
    for j = 1:K

        Xj = X(:, y == j); % all samples in class j

        rand_samples = randperm(size(Xj, 2), L); % choose L unique random sample index from class j
        M(:, (j*L - L + 1) : j*L) = Xj(:, rand_samples); % insert the selected L samples into the correct block of the prototype matrix M
                                                         % since we are taking 4~6 index for class 2 if 1,1,1,2,2,2,3,...
        pro_labels((j*L - L + 1) : j*L) = j;             
    end

    % LVQ training
    for t = 1:N % N iterations for the training

        idx = randi(p); % randomly pick one sample from the training data
        x = X(:, idx);
        real_label = y(idx);
        
        dis = vecnorm(M - x, 2, 1); % compute distances between the sample and all prototypes
        [~, min_idx] = min(dis); % find the closest prototype
        m = M(:, min_idx);
        m_class = pro_labels(min_idx);

        alpha = alphas(t); % decrease the learning rate over time
        
        if m_class == real_label % if the randomly chosen sample and its closest prototype have the same class
            M(:, min_idx) = m + alpha * (x - m); % pull the prototype toward the sample
        else % if NOT
            M(:, min_idx) = m - alpha * (x - m); % push it away from the sample
        end
    end

    % internal evaluation on training data
    y_eval = zeros(1, p);
    for i = 1:p

        xi = X(:, i); % take each samples
        dis = vecnorm(M - xi, 2, 1);
        [~, min_idx] = min(dis); % find the prototype index with minimum distance

        y_eval(i) = pro_labels(min_idx); % assign the label of the nearest prototype
    end

    acc = mean(y_eval == y);
    fprintf('Training Accuracy (L = %d): %.2f%%\n', L, acc * 100);

    % confusion matrix
    figure;
    cm = confusionmat(y, y_eval);
    confusionchart(cm, classes);
    title(sprintf('LVQ Confusion Matrix (L = %d)', L));
end

%%

test_data = readtable("data/forest/testing.csv");
test_labels = categorical(test_data.class);
test_data.class = [];
X_test = table2array(test_data)';
y_test = grp2idx(test_labels)';
p_test = size(X_test, 2);

y_test_eval = zeros(1, p_test);
for i = 1:p_test

    xi = X_test(:, i);
    dis = vecnorm(M - xi, 2, 1);  
    [~, min_idx] = min(dis);

    y_test_eval(i) = pro_labels(min_idx);
end

test_acc = mean(y_test_eval == y_test);
fprintf('Testing Accuracy (L = %d): %.2f%%\n', L, test_acc * 100);

figure;
cm = confusionmat(y_test, y_test_eval);
confusionchart(cm, classes);
title(sprintf('LVQ Confusion Matrix (L = %d)', L));