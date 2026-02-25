% This function computes the boundary of the nucleus mask

function mask = boundaryLoss(Y)
bias = dlarray(0, 'C'); 

% Compute nucleus boundary by Laplacian kernel
laplacianKernel = dlarray([-1 -1 -1; -1 8 -1; -1 -1 -1], 'SS');
boundary = double(abs(dlconv(Y, laplacianKernel, bias, 'Padding', 'same')) > 0);

% Dilate nucleus boundary by dilation kernel
dilationKernel = dlarray(ones(5, 5), 'SS');
mask = double(dlconv(boundary, dilationKernel, bias, 'Padding', 'same') > 0);
