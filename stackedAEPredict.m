function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, dropoutFraction, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.


W1 = stack{1}.w;
W2 = stack{2}.w;
b1 = stack{1}.b;
b2 = stack{2}.b;

a1 = data;
z2 = W1*a1 + repmat(b1, 1, size(data,2));
a2 = sigmoid(z2);
if(dropoutFraction > 0)
   a2 = a2.*(1 - dropoutFraction);
end
z3 = W2*a2 + repmat(b2, 1, size(data,2));
a3 = sigmoid(z3);
if(dropoutFraction > 0)
   a3 = a3.*(1 - dropoutFraction);
end

prob = exp(softmaxTheta*a3);
prob_norm = prob./repmat(sum(prob),numClasses,1);

[tmp,pred]=max(prob_norm);







% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
