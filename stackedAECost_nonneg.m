function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda1, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

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

% W1_neg = zeros(size(W1,1), size(W1,2));
% W2_neg = zeros(size(W2,1), size(W2,2));
% W1_neg(find(W1<0))=W1(find(W1<0));
% W2_neg(find(W2<0))=W2(find(W2<0));
% % weight_neg_decay = sum(sum(W1_neg.^2)) + sum(sum(W2_neg.^2));
% weight_neg_decay = sum(sum(W1_neg.^3)) + sum(sum(W2_neg.^3));
% 
% W1_neg_abs = W1_neg;
% W1_neg_abs(W1_neg_abs~=0)=1;
% 
% W2_neg_abs = W2_neg;
% W2_neg_abs(W2_neg_abs~=0)=1;
% 
% W1_pos = zeros(size(W1,1), size(W1,2));
% W2_pos = zeros(size(W2,1), size(W2,2));
% W1_pos(find(W1>0))=W1(find(W1>0));
% W2_pos(find(W2>0))=W2(find(W2>0));
% weight_pos_decay = sum(sum(W1_pos.^2)) + sum(sum(W2_pos.^2));

idx1 = find(W1 < 0);
idx2 = find(W1 <= -1);
idx3 = find(W1 >= 0);

idx4 = find(W2 < 0);
idx5 = find(W2 <= -1);
idx6 = find(W2 >= 0);

L2_regN = sum(sum(W1(idx1).^2))+sum(sum(W2(idx4).^2));
L2_regP = sum(sum(W1(idx3).^2))+sum(sum(W2(idx6).^2));

% stackgrad{1}.w = delta2*(a1')./size(data,2) - 3/2*lambda*W1_neg.^2 + 0.1*lambda*W1_pos;
% stackgrad{1}.w = delta2*(a1')./(size(data,2)) + lambda*W1_neg - 0.1*lambda*W1_neg_abs;
stackgrad{1}.w = delta2*(a1')./size(data,2);
stackgrad{1}.w(idx1) = stackgrad{1}.w(idx1) + lambda1*W1(idx1);
% stackgrad{1}.w(idx3) = stackgrad{1}.w(idx3) + lambda2*W1(idx3);
stackgrad{1}.b = sum(delta2,2)./size(data,2) ;

% stackgrad{2}.w = delta3*(a2')./size(data,2) - 3/2*lambda*W2_neg.^2 + 0.1*lambda*W2_pos;
% stackgrad{2}.w = delta3*(a2')./(size(data,2)) + lambda*W2_neg - 0.1*lambda*W2_neg_abs;
stackgrad{2}.w = delta3*(a2')./size(data,2);
stackgrad{2}.w(idx4) = stackgrad{2}.w(idx4) + lambda1*W2(idx4);
% stackgrad{2}.w(idx6) = stackgrad{2}.w(idx6) + lambda2*W2(idx6);
stackgrad{2}.b = sum(delta3,2)./size(data,2);

% softmaxTheta_neg = zeros(size(softmaxTheta,1), size(softmaxTheta,2));
% softmaxTheta_neg(find(softmaxTheta<0)) = softmaxTheta(find(softmaxTheta<0));
% softmaxTheta_neg_decay = sum(sum(softmaxTheta_neg.^2)) ;
% 
% softmaxTheta_neg_abs = softmaxTheta_neg;
% softmaxTheta_neg_abs(softmaxTheta_neg_abs~=0)=1;

% softmaxTheta_decay = sum(sum(softmaxTheta.^2)) ;

idx7 = find(softmaxTheta<0);
idx8 = find(softmaxTheta>=0);

softmax_L2_regN = sum(sum(softmaxTheta(idx7).^2));
softmax_L2_regP = sum(sum(softmaxTheta(idx8).^2));

softmaxThetaGrad = -1/size(data,2) * (a3*(groundTruth-prob_norm)') ;

softmaxThetaGrad = softmaxThetaGrad';
softmaxThetaGrad(idx7) = softmaxThetaGrad(idx7) + lambda1*softmaxTheta(idx7);
% softmaxThetaGrad(idx8) = softmaxThetaGrad(idx8) + lambda2*softmaxTheta(idx8);

% softmaxThetaGrad = softmaxThetaGrad'+ lambda*softmaxTheta;
% softmaxThetaGrad = softmaxThetaGrad' + lambda*softmaxTheta_neg - 0.3*lambda*softmaxTheta_neg_abs;

% -------------------------------------------------------------------------

% cost = -sum(sum(groundTruth.*log(prob_norm)))/size(data,2) + lambda/2*weight_neg_decay + lambda/2*softmaxTheta_neg_decay;

% cost = -sum(sum(groundTruth.*log(prob_norm)))/size(data,2) + lambda1/2*softmax_L2_regN;
cost = -sum(sum(groundTruth.*log(prob_norm)))/size(data,2) + lambda1/2*softmax_L2_regN + lambda1/2*L2_regN;
% cost_acc = -sum(sum(groundTruth.*log(prob_norm)))/size(data,2)
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
