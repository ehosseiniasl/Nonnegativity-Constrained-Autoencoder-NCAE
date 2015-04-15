function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda1, data, labels)
                                        


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; 


M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% 

W1 = stack{1}.w;
W2 = stack{2}.w;
b1 = stack{1}.b;
b2 = stack{2}.b;

a1 = data;
z2 = W1*a1 + repmat(b1, 1, size(data,2));
a2 = sigmoid(z2);
z3 = W2*a2 + repmat(b2, 1, size(data,2));
a3 = sigmoid(z3);

prob = exp(softmaxTheta*a3);
[r,c] = find(isinf(prob));
prob(r,c) = exp(709);  % avoid Inf in prob matrix
prob_norm = prob./repmat(sum(prob),numClasses,1);
[r,c] = find(prob_norm == 0);
prob_norm(r,c) = eps;

a4 = prob_norm;

delta3 = -(softmaxTheta'*(groundTruth-prob_norm)) .* (a3.*(ones(size(softmaxTheta,2),size(data,2))-a3));

delta2 = (W2'*delta3) .* (a2.*(ones(size(W2,2),size(data,2))-a2));

delta1 = (W1'*delta2) .* (a1.*(ones(size(W1,2),size(data,2))-a1));

idx1 = find(W1 < 0);
idx2 = find(W1 <= -1);
idx3 = find(W1 >= 0);

idx4 = find(W2 < 0);
idx5 = find(W2 <= -1);
idx6 = find(W2 >= 0);

L2_regN = sum(sum(W1(idx1).^2))+sum(sum(W2(idx4).^2));
L2_regP = sum(sum(W1(idx3).^2))+sum(sum(W2(idx6).^2));


stackgrad{1}.w = delta2*(a1')./size(data,2);
stackgrad{1}.w(idx1) = stackgrad{1}.w(idx1) + lambda1*W1(idx1);
stackgrad{1}.b = sum(delta2,2)./size(data,2) ;

stackgrad{2}.w = delta3*(a2')./size(data,2);
stackgrad{2}.w(idx4) = stackgrad{2}.w(idx4) + lambda1*W2(idx4);
stackgrad{2}.b = sum(delta3,2)./size(data,2);


idx7 = find(softmaxTheta<0);
idx8 = find(softmaxTheta>=0);

softmax_L2_regN = sum(sum(softmaxTheta(idx7).^2));
softmax_L2_regP = sum(sum(softmaxTheta(idx8).^2));

softmaxThetaGrad = -1/size(data,2) * (a3*(groundTruth-prob_norm)') ;

softmaxThetaGrad = softmaxThetaGrad';
softmaxThetaGrad(idx7) = softmaxThetaGrad(idx7) + lambda1*softmaxTheta(idx7);

cost = -sum(sum(groundTruth.*log(prob_norm)))/size(data,2) + lambda1/2*softmax_L2_regN + lambda1/2*L2_regN;

if isnan(cost)
    error()
end

%% Roll gradient vector

grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
