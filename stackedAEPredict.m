function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, dropoutFraction, data)
                                         

 
%% Unroll theta parameter

softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% 

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



end


function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
