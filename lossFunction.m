% This function computes the loss function between the predicted nucleus
% segmentation (Y) and the target nucleus segmentation (T).

function loss = lossFunction(Y,T)
N = size(Y,4);
Y0 = Y;
T0 = T;

% Compute first term of the loss function based on the Intersection over Union (IoU) or Jaccard index
Pcnot = 1-Y;
Gcnot = 1-T;
TP = sum(sum(Y.*T,1),2); % True positives
FP = sum(sum(Y.*Gcnot,1),2); % False positives
FN = sum(sum(Pcnot.*T,1),2); % False negatives
numer = TP + eps;
denom = TP + FP + FN + eps;
loss1tot = 1 - numer./denom; 
IoU = sum(sum(loss1tot,3))/N;

% Compute second term of the loss function based on the boundary Intersection over Union (bIoU)
bmaskY = boundaryLoss(double((-Y(:,:,1,:)+Y(:,:,2,:))>0)); % boundary of the predicted nucleus mask
bmaskT = boundaryLoss(double((-T(:,:,1,:)+T(:,:,2,:))>0)); % boundary of the target nucleus mask
inters = sum(bmaskT.*bmaskY,[1 2]);
union = sum((bmaskT+bmaskY)>0,[1 2]);
loss2tot = 1 - inters./union;
bIoU = sum(sum(loss2tot,3))/N;

% Compute third term of the loss function based on the Euclidean distance (D)
erodeKernel = dlarray(ones(5,5), 'SS');
bias = dlarray(0, 'C'); 
emaskY = double(abs(dlconv(double((-Y0(:,:,1,:)+Y0(:,:,2,:))>0), erodeKernel, bias, 'Padding', 'same')) > sum(sum(erodeKernel))/2); % erosion of the predicted nucleus mask to avoid errors in the initial steps of the training
maskT = double((-T0(:,:,1,:)+T0(:,:,2,:))>0); % target nucleus mask 
[cenx_Y,ceny_Y] = centroidLoss(emaskY); % centroid of the predicted nucleus mask
[cenx_T,ceny_T] = centroidLoss(maskT); % centroid of the target nucleus mask
D = sum(sqrt((cenx_Y-cenx_T).^2+(ceny_Y-ceny_T).^2))/N;

% Compute loss function
lambda1 = 0.1;
lambda2 = 0.1;
lambda3 = 0.8;
loss = lambda1*IoU + lambda2*bIoU + lambda3*D;

