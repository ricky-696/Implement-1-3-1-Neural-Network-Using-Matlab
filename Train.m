% activation function
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmoidGrad = @(x) sigmoid(x) .* (1-sigmoid(x));

data = [-4, -3, -2, 1, -1, 2, 3, 4];
label = [1 1 1 1 0 0 0 0];


learningRate = 0.1;
Iterations = 10000;

% weight & bias
W1 = rand(3, 1);
b1 = rand(3, 1);
W2 = rand(1, 3);
b2 = rand(1, 1);

for i = 1 : Iterations
    % Forward propagation
    Z1 = W1 * data + b1;
    A1 = sigmoid(Z1);
    Z2 = W2 * A1 + b2;
    A2 = sigmoid(Z2);

    loss = sum((A2 - label).^2) / length(label);

    % accuarcy
    acc = zeros(1, 8);
    for j = 1 : size(A2, 2)
        if A2(1, j) >= 0.5
            pred = 1;
        else
            pred = 0;
        end

        if pred == label(1, j)
            acc(1, j) = 1;
        end
    end
    fprintf("Epoch: %d, Loss: %f, Acc: %f\n", i, loss, sum(acc, "all") / size(acc, 2));

    % Backward propagation
    dZ2 = A2 - label;
    dW2 = (dZ2*A1') / length(label);
    db2 = sum(dZ2, 2) / length(label);
    dZ1 = W2'*dZ2.*sigmoidGrad(Z1);
    dW1 = (dZ1*data') / length(label);
    db1 = sum(dZ1, 2) / length(label);

    % learn the weight & bias
    W1 = W1 - learningRate * dW1;
    b1 = b1 - learningRate * db1;
    W2 = W2 - learningRate * dW2;
    b2 = b2 - learningRate * db2;
end

disp('Weight1 & bias1 :');
disp([W1, b1]);
disp('Weight2 & bias2 :');
disp([W2, b2]);
