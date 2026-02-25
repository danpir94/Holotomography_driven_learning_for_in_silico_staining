% This function computes the centroid of the nucleus mask

function [centroid_x,centroid_y] = centroidLoss(Y)

[height, width] = size(Y,[1 2]);
[XX, YY] = meshgrid(1:width, 1:height);
XX = dlarray(XX, 'SS');  
YY = dlarray(YY, 'SS');
total_active_pixels = sum(Y, [1 2]);

centroid_x = sum(XX .* Y, [1 2]) ./ total_active_pixels;
centroid_y = sum(YY .* Y, [1 2]) ./ total_active_pixels;