% This script trains the Holotomography-driven CNN for the in-silico
% staining of the 2D QPMs of single cells recorded in flow cytometry
% conditions avoiding co-registration with fluorescent images.
 
%% Load Training set and Validation set
close all
clear all
clc

D = ['Dataset\Training\Input'];
imds1 = imageDatastore(D);
D = ['Dataset\Training\Target'];
imds2 = pixelLabelDatastore(D,["BG" "N"],[0 255]);
imds2.ReadFcn = @(loc)transformIMG_label(imread(loc));
XTrain = imds1; % Cell QPMs
YTrain = imds2; % Nucleus binary masks

D = ['Dataset\Validation\Input'];
imds1 = imageDatastore(D);
D = ['Dataset\Validation\Target'];
imds2 = pixelLabelDatastore(D,["BG" "N"],[0 255]);
imds2.ReadFcn = @(loc)transformIMG_label(imread(loc));
XVal = imds1; % Cell QPMs
YVal = imds2; % Nucleus binary masks

%% Load CNN architecture
numClasses = 2;
imageSize = [96 96 1];
numFilter = 4;
kernelSize = 3;
lgraph = resnet16(imageSize,numClasses,numFilter,kernelSize);

%% TRAINING

% Create folder where checkpoints are saved
modelDateTime = string(datetime('now','Format',"yyyy-MM-dd-HH-mm-ss"));
fold_Checkpoints = ['Checkpoints\' char(modelDateTime)];
mkdir(fold_Checkpoints)

% Create datastore for training and validation
dsTrain = combine(XTrain,YTrain);
dsVal = combine(XVal,YVal);

% Add a random translation to each image of the training set at each
% iteration to improve generalization
augmenteddsTrain = transform(dsTrain,@translateDatastore);

% Set training options
maxEpochs = 100; 
initLearningRate = 1e-4; 
miniBatchSize = 256; 

options = trainingOptions("adam", ...
    'ExecutionEnvironment',"gpu", ...
    'InitialLearnRate',initLearningRate, ...
    'ValidationData',dsVal, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'shuffle','every-epoch', ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',5, ... % round(length(XTrain.Files)/miniBatchSize) %10
    'Plots','training-progress', ...
    'Verbose',true,...
    'OutputNetwork','best-validation-loss',...
    'CheckpointPath',fold_Checkpoints,...
    'CheckpointFrequency',5,...
    'CheckpointFrequencyUnit','iteration');

% Train and save the Holotomography-driven CNN
[net,info] = trainnet(augmenteddsTrain,lgraph,@lossFunction,options);
save(strcat("trainedNet-",modelDateTime,"-Epoch-",num2str(maxEpochs),".mat"),'net','info');


