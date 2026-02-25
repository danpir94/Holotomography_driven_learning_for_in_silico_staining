% This function builds the architecture of the Holotomography-driven CNN.
% imageSize : size of the input image, specified as [height, width, channels]. 
% numClasses  : number of output classes for segmentation/classification. 
% numFilter   : number of filters in the initial convolutional layers. 
% kernelSize  : size of the square convolutional kernels (filters).

function lgraph = resnet16(imageSize,numClasses,numFilter,kernelSize)

% Parameters
numFilter2 = 2 * numFilter;
numFilter4 = 4 * numFilter;
padSize = (kernelSize-1)/2;

% Initial layer graph
lgraph = dlnetwork;

% Input layer
tempNet = imageInputLayer(imageSize,"Name","newInputLayer","Normalization","rescale-zero-one","NormalizationDimension","channel");
lgraph = addLayers(lgraph,tempNet);

% Initial Conv + Pooling
tempNet = [
    convolution2dLayer([7 7],numFilter,"Name","newConv1","Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","bn_conv1")
    reluLayer("Name","conv1_relu")
    maxPooling2dLayer([3 3],"Name","pool1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempNet);

% First Residual Block (res2a)
tempNet = [
    convolution2dLayer([kernelSize kernelSize],numFilter,"Name","res2a_branch2a","Padding",[padSize padSize padSize padSize])
    batchNormalizationLayer("Name","bn2a_branch2a")
    reluLayer("Name","res2a_branch2a_relu")
    convolution2dLayer([kernelSize kernelSize],numFilter,"Name","res2a_branch2b","Padding",[padSize padSize padSize padSize])
    batchNormalizationLayer("Name","bn2a_branch2b")];
lgraph = addLayers(lgraph,tempNet);

tempNet = [
    additionLayer(2,"Name","res2a")
    reluLayer("Name","res2a_relu")];
lgraph = addLayers(lgraph,tempNet);

% Second Residual Block (res2b)
tempNet = [
    convolution2dLayer([kernelSize kernelSize],numFilter,"Name","res2b_branch2a","Padding",[padSize padSize padSize padSize])
    batchNormalizationLayer("Name","bn2b_branch2a")
    reluLayer("Name","res2b_branch2a_relu")
    convolution2dLayer([kernelSize kernelSize],numFilter,"Name","res2b_branch2b","Padding",[padSize padSize padSize padSize])
    batchNormalizationLayer("Name","bn2b_branch2b")];
lgraph = addLayers(lgraph,tempNet);

tempNet = [
    additionLayer(2,"Name","res2b")
    reluLayer("Name","res2b_relu")];
lgraph = addLayers(lgraph,tempNet);

% Downsampling Residual Block (res3a)
tempNet = [
    convolution2dLayer([kernelSize kernelSize],numFilter2,"Name","res3a_branch2a","Padding",[padSize padSize padSize padSize],"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch2a")
    reluLayer("Name","res3a_branch2a_relu")
    convolution2dLayer([kernelSize kernelSize],numFilter2,"Name","res3a_branch2b","Padding",[padSize padSize padSize padSize])
    batchNormalizationLayer("Name","bn3a_branch2b")];
lgraph = addLayers(lgraph,tempNet);

tempNet = [
    convolution2dLayer([1 1],numFilter2,"Name","res3a_branch1","Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch1")];
lgraph = addLayers(lgraph,tempNet);

tempNet = [
    additionLayer(2,"Name","res3a")
    reluLayer("Name","res3a_relu")];
lgraph = addLayers(lgraph,tempNet);

% Additional residual block (res3b)
tempNet = [
    convolution2dLayer([kernelSize kernelSize],numFilter2,"Name","res3b_branch2a","Padding",[padSize padSize padSize padSize])
    batchNormalizationLayer("Name","bn3b_branch2a")
    reluLayer("Name","res3b_branch2a_relu")
    convolution2dLayer([kernelSize kernelSize],numFilter2,"Name","res3b_branch2b","Padding",[padSize padSize padSize padSize])
    batchNormalizationLayer("Name","bn3b_branch2b")];
lgraph = addLayers(lgraph,tempNet);

tempNet = [
    additionLayer(2,"Name","res3b")
    reluLayer("Name","res3b_relu")];
lgraph = addLayers(lgraph,tempNet);

% ASPP modules with different dilation rates
tempNet = [
    convolution2dLayer([1 1],numFilter4,"Name","aspp_Conv_1","Padding","same")
    batchNormalizationLayer("Name","aspp_BatchNorm_1")
    reluLayer("Name","aspp_Relu_1")];
lgraph = addLayers(lgraph,tempNet);

tempNet = [
    convolution2dLayer([kernelSize kernelSize],numFilter4,"Name","aspp_Conv_2","DilationFactor",[6 6],"Padding","same")
    batchNormalizationLayer("Name","aspp_BatchNorm_2")
    reluLayer("Name","aspp_Relu_2")];
lgraph = addLayers(lgraph,tempNet);

tempNet = [
    convolution2dLayer([kernelSize kernelSize],numFilter4,"Name","aspp_Conv_3","DilationFactor",[12 12],"Padding","same")
    batchNormalizationLayer("Name","aspp_BatchNorm_3")
    reluLayer("Name","aspp_Relu_3")];
lgraph = addLayers(lgraph,tempNet);

tempNet = [
    convolution2dLayer([kernelSize kernelSize],numFilter4,"Name","aspp_Conv_4","DilationFactor",[18 18],"Padding","same")
    batchNormalizationLayer("Name","aspp_BatchNorm_4")
    reluLayer("Name","aspp_Relu_4")];
lgraph = addLayers(lgraph,tempNet);

% Decoder path
tempNet = [
    depthConcatenationLayer(4,"Name","catAspp")
    convolution2dLayer([1 1],numFilter4,"Name","dec_c1")
    batchNormalizationLayer("Name","dec_bn1")
    reluLayer("Name","dec_relu1")
    transposedConv2dLayer([8 8],numFilter4,"Name","dec_upsample1","Cropping",[2 2 2 2],"Stride",[4 4])];
lgraph = addLayers(lgraph,tempNet);

tempNet = [
    convolution2dLayer([1 1],3,"Name","dec_c2")
    batchNormalizationLayer("Name","dec_bn2")
    reluLayer("Name","dec_relu2")];
lgraph = addLayers(lgraph,tempNet);

% Crop to match size
tempNet = crop2dLayer("centercrop","Name","dec_crop1");
lgraph = addLayers(lgraph,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","dec_cat1")
    convolution2dLayer([kernelSize kernelSize],numFilter4,"Name","dec_c3","Padding","same")
    batchNormalizationLayer("Name","dec_bn3")
    reluLayer("Name","dec_relu3")
    convolution2dLayer([kernelSize kernelSize],numFilter4,"Name","dec_c4","Padding","same")
    batchNormalizationLayer("Name","dec_bn4")
    reluLayer("Name","dec_relu4")
    convolution2dLayer([1 1],numClasses,"Name","scorer")
    transposedConv2dLayer([8 8],numClasses,"Name","dec_upsample2","Cropping",[2 2 2 2],"Stride",[4 4])];
lgraph = addLayers(lgraph,tempNet);

tempNet = [
    crop2dLayer("centercrop","Name","dec_crop2")
    softmaxLayer("Name","softmax-out")];
lgraph = addLayers(lgraph,tempNet);

% clean up helper variable
clear tempNet;

% Connections
lgraph = connectLayers(lgraph,"newInputLayer","newConv1");
lgraph = connectLayers(lgraph,"newInputLayer","dec_crop2/ref");
lgraph = connectLayers(lgraph,"pool1","res2a_branch2a");
lgraph = connectLayers(lgraph,"pool1","res2a/in2");
lgraph = connectLayers(lgraph,"bn2a_branch2b","res2a/in1");
lgraph = connectLayers(lgraph,"res2a_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"res2a_relu","res2b/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2b","res2b/in1");
lgraph = connectLayers(lgraph,"res2b_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"res2b_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"res2b_relu","dec_c2");
lgraph = connectLayers(lgraph,"bn3a_branch2b","res3a/in1");
lgraph = connectLayers(lgraph,"bn3a_branch1","res3a/in2");
lgraph = connectLayers(lgraph,"res3a_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"res3a_relu","res3b/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2b","res3b/in1");
lgraph = connectLayers(lgraph,"res3b_relu","aspp_Conv_1");
lgraph = connectLayers(lgraph,"res3b_relu","aspp_Conv_2");
lgraph = connectLayers(lgraph,"res3b_relu","aspp_Conv_3");
lgraph = connectLayers(lgraph,"res3b_relu","aspp_Conv_4");
lgraph = connectLayers(lgraph,"aspp_Relu_1","catAspp/in1");
lgraph = connectLayers(lgraph,"aspp_Relu_2","catAspp/in2");
lgraph = connectLayers(lgraph,"aspp_Relu_3","catAspp/in3");
lgraph = connectLayers(lgraph,"aspp_Relu_4","catAspp/in4");
lgraph = connectLayers(lgraph,"dec_upsample1","dec_crop1/in");
lgraph = connectLayers(lgraph,"dec_relu2","dec_crop1/ref");
lgraph = connectLayers(lgraph,"dec_relu2","dec_cat1/in1");
lgraph = connectLayers(lgraph,"dec_crop1","dec_cat1/in2");
lgraph = connectLayers(lgraph,"dec_upsample2","dec_crop2/in");

% Initialization
lgraph = initialize(lgraph);