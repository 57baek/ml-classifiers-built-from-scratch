%%

clc, clear, close all

data = readtable("data/forest/training.csv"); % currently in row-based (samples by features)
%data = readtable("data/forest/testing.csv");

labels = categorical(data.class); % convert labels from cell to category
data.class = []; % remove the non-numeric column
features = table2array(data); % convert remaining numeric part

% summary(labels) % Tells no undefined
% summary(features) % Tells no NA

% transpose to column-based: features by samples
X = features'; % Matrix of data (features by samples)
y = labels'; % Vector of labels (or outputs)
[n, p] = size(X);  % n = number of features, p = number of samples

classes = categories(y);
K = numel(classes);
y_idx = grp2idx(y)'; % convert each class into numeric index
                     % by default, grp2idx(y) returns an n by 1 column vector -> need to transpose

%%

% range of principal components to test
m_range = 1:n; % max = 27 since data = 198 by 27
acc = zeros(size(m_range));

max_acc = 0;
best_m = 0;
best_P = cell(K,1);

% looping over the number of principal components or subspace dimensions m (same as rank d in low-rank SVD)
for m_idx = m_range
    
    m = m_idx;
    P = cell(K,1); % Projection matrices for each class: P{1,2,3,4}

    % PCA subspaces for each class 
    for j = 1:K
        Xj = X(:, y_idx == j); % select ONLY the samples belonging to class j
        [U, ~, ~] = svd(Xj, 'econ'); % compute SVD of class j data matrix
        Uj = U(:, 1:m); % take the top-m (same as rank d for low-rank SVD) left singular vectors (principal directions)
        P{j} = Uj * Uj'; % construct the orthogonal projector onto the m-dimensional PCA subspace of class j
    end

    % classify samples based on projection error
    y_argmin = zeros(1, p); % store predicted class for each sample
    
    % looping over samples of each subspace dimension m (the number of principal components)
    for i = 1:p
        x = X(:, i); % column i = sample i (27 by 1)
        errors = zeros(1, K); % error table since we need to get the class that returns the least error
        % looping over classes of each sample
        for j = 1:K
            pjx = P{j} * x; % project onto class j subspace
            errors(j) = norm(x - pjx); % compute projection error
        end
        [~, argmin_class] = min(errors); % choose class with lowest error (argmin)
        y_argmin(i) = argmin_class; % append predicted label
    end

    % compute accuracy
    acc(m_idx) = mean(y_argmin == y_idx);

    if acc(m_idx) > max_acc
        max_acc = acc(m_idx);
        best_m = m;
        best_P = P;   % Save the winning projection matrices
    end

end

%%

% plot accuracy vs. number of principal components
figure;
plot(m_range, acc * 100, '-ob', 'LineWidth', 2);
xlabel('Number of Principal Components (m)', FontSize=15);
ylabel('Training Accuracy (%)', FontSize=15);
title('PCA Classifier Accuracy vs. m for the Training Dataset', FontSize=15);
grid on;

%%

test_data = readtable("data/forest/testing.csv");
test_labels = categorical(test_data.class);
test_data.class = [];
test_features = table2array(test_data);

X_test = test_features';        
y_test_idx = grp2idx(test_labels)'; 
p_test = size(X_test, 2);      

y_test_pred = zeros(1, p_test);
for i = 1:p_test
    x = X_test(:, i);
    errors = zeros(1, K);
    for j = 1:K
        pjx = best_P{j} * x;
        errors(j) = norm(x - pjx);
    end
    [~, argmin_class] = min(errors);
    y_test_pred(i) = argmin_class;
end

test_accuracy = mean(y_test_pred == y_test_idx);
fprintf('Test Accuracy with m = %d: %.2f%%\n', best_m, test_accuracy * 100);


%%

cm = confusionmat(y_test_idx, y_test_pred);

figure;
confusionchart(cm, classes);
title(['Confusion Matrix for PCA Classifier (m = ' num2str(best_m) ')']);






















%% Does the training dataset really give the best Pj and m

max_acc = 0;
max_m = 0;

for best_m = 1:n
    P = cell(K, 1);
    for j = 1:K
        Xj = X(:, y_idx == j);
        [U, ~, ~] = svd(Xj, 'econ');
        Uj = U(:, 1:best_m);
        P{j} = Uj * Uj';
    end
    
    test_data = readtable("data/forest/testing.csv");
    test_labels = categorical(test_data.class);
    test_data.class = [];
    test_features = table2array(test_data);
    
    X_test = test_features';      
    y_test_idx = grp2idx(test_labels)'; 
    p_test = size(X_test, 2);
    
    y_test_pred = zeros(1, p_test);
    for i = 1:p_test
        x = X_test(:, i);
        errors = zeros(1, K);
        for j = 1:K
            pjx = P{j} * x;
            errors(j) = norm(x - pjx);
        end
        [~, argmin_class] = min(errors);
        y_test_pred(i) = argmin_class;
    end
    
    test_accuracy = mean(y_test_pred == y_test_idx);
    fprintf('Test Accuracy with m = %d: %.2f%%\n', best_m, test_accuracy * 100);

    if test_accuracy > max_acc
        max_acc = test_accuracy;
        max_m = best_m;
    end
end

fprintf('Best Accuracy with m = %d: %.2f%%\n', max_m, max_acc * 100);
