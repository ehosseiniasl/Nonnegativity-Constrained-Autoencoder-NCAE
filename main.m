
clc
clear all
close all

%% Initialize Deep Network Parameters

inputSize = 784;
numClasses = 10;
hiddenSizeL1 = 196;    % Layer 1 Hidden Size
hiddenSizeL2 = 20;    % Layer 2 Hidden Size
sparsityParam = 0.05;   % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter      
beta = 3;              % weight of sparsity penalty term       

inputZeroMaskedFraction   = 0.0;  % denoising ratio
dropoutFraction   = 0.0;          % dropout ratio

%% Load data from the MNIST database

% Load MNIST database files
addpath('/Datasets/MNIST')
trainData = loadMNISTImages('mnist/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

testData = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
testLabels(testLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1


%% STEP 2: Train the first sparse autoencoder


% Randomly initialize the parameters

seed = 1;
sae1Theta = initializeParameters_nonneg(hiddenSizeL1, inputSize, seed);


addpath minFunc/
options.Method = 'lbfgs'; 
options.maxIter = 400;	  
options.display = 'on';



[sae1OptTheta, cost, costhistoty] = minFunc( @(p) sparseAutoencoderCost_nonneg(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, inputZeroMaskedFraction,...
                                   dropoutFraction, sparsityParam, ...
                                   beta, trainData), ...
                                   sae1Theta, options);

%% Train the second sparse autoencoder

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, dropoutFraction, trainData);
                                    
%  Randomly initialize the parameters
sae2Theta = initializeParameters_nonneg(hiddenSizeL2, hiddenSizeL1, seed);

[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost_nonneg(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, inputZeroMaskedFraction,...
                                   dropoutFraction, sparsityParam, ...
                                   beta, sae1Features), ...
                                   sae2Theta, options);

%% Train the softmax classifier

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, dropoutFraction, sae1Features);

%  Randomly initialize the parameters
rand('state',seed);
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);

addpath softmax/

options.maxIter = 100;
softmaxModel = softmaxTrain_nonneg(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainLabels, options);

saeSoftmaxOptTheta = softmaxModel.optTheta(:);


%% Finetune softmax model


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


%% Test 


[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, dropoutFraction, testData);

acc_before(seed) = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc_before(seed) * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, dropoutFraction, testData);

acc_after(seed) = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc_after(seed) * 100);

