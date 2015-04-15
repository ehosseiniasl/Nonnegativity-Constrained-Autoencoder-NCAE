%% CS294A/CS294W Stacked Autoencoder Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sstacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. You will need the initializeParameters.m
%  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
%  
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%

clc
clear all
close all
cd('C:\Users\Ehsan\Dropbox\Codes\stackedae_exercise')
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize = 784;
numClasses = 10;
hiddenSizeL1 = 196;    % Layer 1 Hidden Size
hiddenSizeL2 = 20;    % Layer 2 Hidden Size
sparsityParam = 0.05;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter      
% lambdaL2 = 3e-3;         % weight decay parameter     
% lambdaL3 = 3e-3;         % weight decay parameter     
beta = 3;              % weight of sparsity penalty term       

inputZeroMaskedFraction   = 0.0;
dropoutFraction   = 0.0;

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load MNIST database files
addpath('C:\Users\Ehsan\Desktop\Datasets\MNIST')
trainData = loadMNISTImages('mnist/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

testData = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
testLabels(testLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

addpath('C:\Users\Ehsan\Dropbox\Codes\autoencoder\sparseae_exercise\starter')
addpath('C:\Users\Ehsan\Dropbox\Codes\autoencoder\sparseae_exercise\starter\minFunc')
addpath C:\Users\Ehsan\Dropbox\Codes\minConf\minConf
addpath C:\Users\Ehsan\Dropbox\Codes\minConf\minConf\minConf

%%

for seed = 1:1
    
%  Randomly initialize the parameters
sae1Theta = initializeParameters_nonneg(hiddenSizeL1, inputSize, seed);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta


addpath minFunc/
options.Method = 'lbfgs'; 
options.maxIter = 400;	  
options.display = 'on';



[sae1OptTheta, cost, costhistoty] = minFunc( @(p) sparseAutoencoderCost_nonneg(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, inputZeroMaskedFraction,...
                                   dropoutFraction, sparsityParam, ...
                                   beta, trainData), ...
                                   sae1OptTheta, options);

%---------------------------------------------------------
% options.Method = 'lbfgs'; 
% options.maxIter = 10;	  
% options.display = 'on';
% 
% batch = 100;
% n_batch = size(trainData,2)/batch;
% n_epoch = 1;
% 
% 
% for j = 1:n_epoch
%     
%     for i = 1:n_batch
%         
%         batch_data = trainData(:, (i-1)*batch+1:i*batch);
%         
%         [sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost_nonneg(p, ...
%                                    inputSize, hiddenSizeL1, ...
%                                    lambda, sparsityParam, ...
%                                    beta, batch_data), ...
%                                    sae1Theta, options);
%         sae1Theta = sae1OptTheta;
%     end
% end

% -------------------------------------------------------------------------

% LB = zeros(numel(sae1Theta),1);
% UB = inf(numel(sae1Theta),1);
% 
% batch = 100;
% n_batch = size(trainData,2)/batch;
% n_epoch = 10;
% 
% for j = 1:n_epoch
%     
%     for i = 1:n_batch
%         
%         batch_data = trainData(:, (i-1)*batch+1:i*batch);
%         
%         [sae1OptTheta, cost] = minConf_TMP(@(p) sparseAutoencoderCost_nonneg(p, inputSize, hiddenSizeL1, ...
%             lambda, sparsityParam, ...
%             beta, batch_data), ...
%             sae1Theta, LB, UB, options);
%         sae1Theta = sae1OptTheta;
%     end
% end

                                          % LB = zeros(numel(theta),1);
% UB = inf(numel(theta),1);
% [opttheta, cost] = minConf_TMP(@(p) sparseAutoencoderCost(p, ...
%                                    visibleSize, hiddenSize, ...
%                                    lambda, sparsityParam, ...
%                                    beta, patches), ...
%                                    theta, LB, UB, options);

%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

addpath('C:\Users\Ehsan\Dropbox\Codes\self-thaught learning\stl_exercise')

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, dropoutFraction, trainData);
                                    
%  Randomly initialize the parameters
sae2Theta = initializeParameters_nonneg(hiddenSizeL2, hiddenSizeL1, seed);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta
% 
[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost_nonneg(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, inputZeroMaskedFraction,...
                                   dropoutFraction, sparsityParam, ...
                                   beta, sae1Features), ...
                                   sae2Theta, options);

% -------------------------------------------------------------------------

% for j = 1:n_epoch
%     
%     for i = 1:n_batch
%         
%         batch_sae1Features = sae1Features(:, (i-1)*batch+1:i*batch);
%         [sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost_nonneg(p, ...
%                                    hiddenSizeL1, hiddenSizeL2, ...
%                                    lambda, sparsityParam, ...
%                                    beta, batch_sae1Features), ...
%                                    sae2Theta, options);
%                                
%                                sae2Theta = sae2OptTheta;
%                                
%     end
% end

                               

%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, dropoutFraction, sae1Features);

rand('state',seed);
%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);
% saeSoftmaxTheta = 0.005 * rand(hiddenSizeL2 * numClasses, 1);


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

addpath('C:\Users\Ehsan\Dropbox\Codes\softmax_exercise')

options.maxIter = 100;
softmaxModel = softmaxTrain_nonneg(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainLabels, options);

saeSoftmaxOptTheta = softmaxModel.optTheta(:);

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta{seed} = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%


%% Check Gradient


 checkStackedAECost_nonneg()

%% Fine-tuning AE

options.Method = 'lbfgs'; 
options.maxIter = 400;	  
options.display = 'on';

dbstop if error
[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost_nonneg(p, inputSize, hiddenSizeL2, ...
                                              numClasses, netconfig, ...
                                              lambda, trainData, trainLabels), ...
                                              stackedAETheta, options);

% LB = zeros(numel(stackedAETheta),1);
% UB = inf(numel(stackedAETheta),1);
% [stackedAEOptTheta, cost] = minConf_TMP(@(p) stackedAECost_nonneg(p, inputSize, hiddenSizeL2, ...
%                                               numClasses, netconfig, ...
%                                               lambda, trainData, trainLabels), ...
%                                               stackedAETheta, LB, UB, options);
                                          
%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
% testData = loadMNISTImages('t10k-images.idx3-ubyte');
% testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
% 
% testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, dropoutFraction, testData);

acc_before(seed) = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc_before(seed) * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, dropoutFraction, testData);

acc_after(seed) = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc_after(seed) * 100);

end

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
