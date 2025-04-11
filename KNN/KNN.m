%%

clc; clear; close all;

%data = readtable("data/forest/training.csv");
data = readtable("data/forest/testing.csv");

labels = categorical(data.class);
classes = categories(labels);
data.class = [];
X = table2array(data)';
y = grp2idx(labels)';
[n, p] = size(X);


%% 
k_range = 1:n;
acc = zeros(size(k_range));

best_acc = 0;
best_k = 0;
y_best_pred = zeros(1, p);

for k_idx = 1:length(k_range)
    
    k = k_range(k_idx);

    y_pred = zeros(1, p);
    for i = 1:p
        xi = X(:, i);
        % Compute distances to all other training samples
        dists = vecnorm(X - xi, 2, 1);
        dists(i) = inf; % exclude self in k-NN
        [~, sorted_idx] = sort(dists);
        k_neighbors = y(sorted_idx(1:k));
        y_pred(i) = mode(k_neighbors);
    end
    acc(k_idx) = mean(y_pred == y);

    if acc(k_idx) > best_acc
        best_acc = acc(k_idx);
        best_k = k_idx;
        y_best_pred = y_pred;
    end
end

%% 
figure;
plot(k_range, acc * 100, '-ob');
xlabel('k (number of neighbors)', FontSize=15);
ylabel('Accuracy (%)', FontSize=15);
title('k-NN Classifier Accuracy vs. k for the Training Dataset', FontSize=15);
legend('Location', 'best');
grid on;

%% 

cm = confusionmat(y, y_best_pred);

figure;
confusionchart(cm, classes);
title(['Confusion Matrix for KNN Classifier (m = ' num2str(best_k) ')']);