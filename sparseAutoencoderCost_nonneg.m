function [cost,grad, objhistory] = sparseAutoencoderCost_nonneg(theta, visibleSize, hiddenSize, ...
                                             lambda, inputZeroMaskedFraction, dropoutFraction, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 
objhistory = [];

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

y = data;
a1 = data;
if (inputZeroMaskedFraction>0)
    a1 = a1.*(rand(size(a1))>inputZeroMaskedFraction);
end

z2 = W1*a1 + repmat(b1,1,size(a1,2));
a2 = sigmoid(z2);
%dropout
if(dropoutFraction > 0)
    dropOutMask = (rand(size(a2))>dropoutFraction);
    a2 = a2.*dropOutMask;
end

z3 = W2*a2 + repmat(b2,1,size(a2,2));
a3 = sigmoid(z3);

yhat = a3;

delta3 = -(y - yhat) .* (a3.*(ones(visibleSize,size(y,2))-a3));

% delta2 = W2'*delta3 .* (a2.*(ones(hiddenSize,size(y,2))-a2));


param = sum(a2,2)./size(y,2);
par = sparsityParam*ones(hiddenSize,1);
sparsity = beta*(-par./param + (ones(hiddenSize,1)-par)./(ones(hiddenSize,1)-param));
sparsity = repmat(sparsity,1,size(data,2));

delta2 = (W2'*delta3 + sparsity) .* (a2.*(ones(hiddenSize,size(y,2))-a2));

if(dropoutFraction > 0)
    delta2 = delta2.*dropOutMask;
end

% kl = 0;
% for i = 1:hiddenSize
%     tmp = kl_div(sparsityParam, param(i));
%     kl = kl+tmp;
% end
kl = sum(sparsityParam*log(par./param) + (1-sparsityParam)*log((ones(hiddenSize,1)-par)./(ones(hiddenSize,1)-param)));

% weight_decay = sum(sum(W1.^2)) + sum(sum(W2.^2));

idx1 = find(W1 < 0);
idx2 = find(W1 <= -1);
idx3 = find(W1 >= 0);

idx4 = find(W2 < 0);
idx5 = find(W2 <= -1);
idx6 = find(W2 >= 0);

L2_regN = sum(sum(W1(idx1).^2)) + sum(sum(W2(idx4).^2));
L2_regP = sum(sum(W1(idx3).^2)) + sum(sum(W2(idx6).^2));
L1_reg = sum(abs(W1(:))) + sum(abs(W2(:)));


cost = 0.5*sum(sum((y-yhat).^2))./size(y,2) + beta*kl + lambda/2*L2_regN;
% cost = 0.5*sum(sum((y-yhat).^2))./size(y,2) + beta*kl + lambda1*L1_reg;

newobj = 0.5*sum(sum((y-yhat).^2))./size(y,2);
objhistory = [objhistory newobj];

one1 = ones(size(W1));
one2 = ones(size(W2));

W1grad = delta2*(a1')./(size(y,2));
W1grad(idx1) = W1grad(idx1) + lambda*W1(idx1);
% W1grad(idx1) = W1grad(idx1) - lambda1*one1(idx1);
% W1grad(idx3) = W1grad(idx3) + lambda1*one1(idx3);

 
W2grad = delta3*(a2')./(size(y,2));
W2grad(idx4) = W2grad(idx4) + lambda*W2(idx4);
% W2grad(idx4) = W2grad(idx4) - lambda1*one2(idx4);
% W2grad(idx6) = W2grad(idx6) + lambda1*one2(idx6);


b1grad = sum(delta2,2)./(size(y,2));

b2grad = sum(delta3,2)./(size(y,2));





%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

