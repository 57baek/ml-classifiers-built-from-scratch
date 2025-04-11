%% 

clc; clear; close all;

data = readtable("data/forest/training.csv");
%data = readtable("data/forest/testing.csv");

labels = categorical(data.class);
classes = categories(labels);
K = numel(classes);

data.class = [];
X = table2array(data)'; 
y = grp2idx(labels)';   

[n, p] = size(X);       

%% 

mu_class = zeros(n, K);

S_W = zeros(n, n); % within-class scatter matrix
for j = 1:K % loops over each class

    Xj = X(:, y == j); % all samples that are class j
    mu_j = mean(Xj, 2); % mean for each row (feature) for samples in class j
    mu_class(:, j) = mu_j;
    
    Sj = (Xj - mu_j) * (Xj - mu_j)'; % computes the scatter matrix for class j
    S_W = S_W + Sj; % the SUM within-class scatter matrix
end

mu_all = mean(X, 2); % the mean of a feature (2 = row) across all samples

S_B = zeros(n, n); % between-class scatter matrix 
for j = 1:K

    pj = sum(y == j); % how many samples belong to class j
    mu_j = mu_class(:, j); % mean for each row (feature) for samples in class j

    S_B = S_B + pj * ((mu_j - mu_all) * (mu_j - mu_all)');
end

[W, D] = eig(S_B, S_W);
[~, idx] = sort(diag(D), 'descend');
W = W(:, idx); % sort eigenvectors by the largest eigenvalue order
Q = W(:, 1:K-1); % take the top K-1 eigenvectors to form the projection matrix Q 
                 % at most K - 1 useful discriminant directions

Z = Q' * X;

centroids = zeros(K-1, K);
for j = 1:K
    Zj = Z(:, y == j);
    centroids(:, j) = mean(Zj, 2);
end

%% 

figure;
gscatter(Z(1, :), Z(2, :), y, 'rgbm', 'o', 'filled'); 
xlabel('LDA 1');
ylabel('LDA 2');
title('LDA Projection of Training Data', FontSize=15);
legend(classes, 'Location', 'best'); 
grid on;

%%

figure;
scatter3(Z(1, :), Z(2, :), Z(3, :), 50, y, 'filled');
xlabel('LDA 1');
ylabel('LDA 2');
zlabel('LDA 3');
title('3D LDA Projection of Training Data', FontSize=15);
legend(classes, 'Location', 'best');
grid on;





%% 

test_data = readtable("data/forest/testing.csv");
test_labels = categorical(test_data.class);
y_test = grp2idx(test_labels)';
test_data.class = [];

X_test = table2array(test_data)';
[~, p_test] = size(X_test);       

Z_test = Q' * X_test; 

y_pred = zeros(1, p_test);

for j = 1:p_test
    z = Z_test(:, j); 
    dis = vecnorm(centroids - z, 2, 1); % compute L2 distances to each class centroid
    [~, y_pred(j)] = min(dis); % assign the closest centroid's class label for each sample
end

acc = mean(y_pred == y_test);

fprintf('LDA Classifier Test Accuracy: %.2f%%\n', acc * 100);

%% 

figure;
confusionchart(confusionmat(y_test, y_pred), classes);
title('Confusion Matrix (LDA Classifier)');

