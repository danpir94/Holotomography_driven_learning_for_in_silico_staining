% This script calculates the F1 score and the IoU metrics about QPMs
% reported in the paper in Fig. 3. Moreover, all the input, target,
% and predicted images of Fig. 3 can be seen about the training,
% validation, internal test, independent internal test, and independent
% external test sets.

%% Load Trained Network

close all
clear all
clc

name = 'trainedNet-2025-07-04-11-25-00-Epoch-100'; % type name of the trained network. If you want to reproduce exactly the same results of the paper, type 'trainedNet_paper'.
load([name '.mat'],'net')

isShow = 0; % type 1 if you want to see images, otherwise 0 if you just want to calculate metrics.

%% Training Set
D = ['Dataset\Training\Input'];
imds1 = imageDatastore(D);
D = ['Dataset\Training\Target'];
imds2 = pixelLabelDatastore(D,["BG" "N"],[0 255]);
imds2.ReadFcn = @(loc)transformIMG_label(imread(loc));
XX = imds1;
YY = imds2;

if isShow == 1
    figure('units','normalized','outerposition',[0 0 1 1])
    sgtitle('     Training Set - HeLa Cells','Fontsize',36)
    s1 = subplot(2,2,1);
    s2 = subplot(2,2,2);
    s3 = subplot(2,2,3);
    s4 = subplot(2,2,4);
    tt1 = [];
    tt2 = [];
    tt3 = [];
end
F1 = zeros(length(XX.Files),1);
IoU = zeros(length(XX.Files),1);
N = 7;
kernel = ones(N, N) / N^2;
for rr = 1:length(XX.Files)
    rr
    x = readimage(XX,rr);
    y = readimage(YY,rr);
    BW2 = double(y)-1; % target nucleus mask
    z = (predict(net,x));
    BW1 = double(convn(double(diff(z,[],3)>0), kernel, 'same')>0.5); % predicted nucleus mask with shape smoothing 
    TP = nnz(BW1 == 1 & BW2 == 1); % true positives
    FP = nnz(BW2 == 0 & BW1 == 1); % false positives
    FN = nnz(BW2 == 1 & BW1 == 0); % false negatives
    F1(rr) = TP/(TP+0.5*FN+0.5*FP);
    IoU(rr) = TP/(TP+FN+FP);
    if isShow == 1
        delete(tt1)
        delete(tt2)
        delete(tt3)

        subplot(s1)
        cla(s1)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(x,2));axis image;axis off;
        set(gca,'Fontsize',15)
        colormap(s1,'gray')
        M = max(x(:));
        caxis([0 round(M)])
        hh = colorbar('Position',[0.395045517502267,0.545086120914231,0.011204481792717,0.285714282994233]);
        xlabel(hh,'[rad]')
        set(hh,'XTick',[0:2:round(M)])
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['INPUT'],['Stain-Free QPM']},'Color',[64 64 64]/255,'Fontsize',18)
    
    
        subplot(s2)
        cla(s2)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(x,2));axis image;axis off;
        pp = bwboundaries(imresize(BW1,2)>0.5);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'-','Linewidth',4,'Color',[0 0.5 0])
        pp = bwboundaries(imresize(BW2,2)>0.5);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'-r','Linewidth',4)
        set(gca,'Fontsize',15)
        colormap(s2,'gray')
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['MERGE'],['In-Silico Stained QPM']},'Color',[0 112 192]/255,'Fontsize',18)
    
        subplot(s3)
        cla(s3)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(BW2,2)>0.5);axis image;axis off;
        BWcell = double(imresize(x>0,2)>0.5);
        pp = bwboundaries(BWcell);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'b--','Linewidth',4)
        colormap(s3,[0 0 0;1 0 0])
        set(gca,'Fontsize',15)
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['TARGET'],['Reprojected Nucleus Mask']},'Color',[255 0 0]/255,'Fontsize',18)
    
    
        subplot(s4)
        cla(s4)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(BW1,2)>0.5);axis image;axis off;
        BWcell = double(imresize(x>0,2)>0.5);
        pp = bwboundaries(BWcell);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'b--','Linewidth',4)
        colormap(s4,[0 0 0;0 0.5 0])
        set(gca,'Fontsize',15)
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['OUTPUT'],['Predicted Nucleus Mask']},'Color',[0 128 0]/255,'Fontsize',18)

        tt1 = annotation('textbox', [0.275 0.20 0.5 0.5], 'String', ['Image = ' num2str(rr)], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        tt2 = annotation('textbox', [0.275 0.10 0.5 0.5], 'String', ['F1 Score = ' num2str(F1(rr),'%.3f')], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        tt3 = annotation('textbox', [0.275 0 0.5 0.5], 'String', ['IoU = ' num2str(IoU(rr),'%.3f')], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        drawnow
    end
end

F1_Training = F1;
IoU_Training = IoU;


%% Validation Set
D = ['Dataset\Validation\Input'];
imds1 = imageDatastore(D);
D = ['Dataset\Validation\Target'];
imds2 = pixelLabelDatastore(D,["BG" "N"],[0 255]);
imds2.ReadFcn = @(loc)transformIMG_label(imread(loc));
XX = imds1;
YY = imds2;

if isShow == 1
    figure('units','normalized','outerposition',[0 0 1 1])
    sgtitle('     Validation Set - HeLa Cells','Fontsize',36)
    s1 = subplot(2,2,1);
    s2 = subplot(2,2,2);
    s3 = subplot(2,2,3);
    s4 = subplot(2,2,4);
    tt1 = [];
    tt2 = [];
    tt3 = [];
end
F1 = zeros(length(XX.Files),1);
IoU = zeros(length(XX.Files),1);
N = 7;
kernel = ones(N, N) / N^2;
for rr = 1:length(XX.Files)
    rr
    x = readimage(XX,rr);
    y = readimage(YY,rr);
    BW2 = double(y)-1; % target nucleus mask
    z = (predict(net,x));
    BW1 = double(convn(double(diff(z,[],3)>0), kernel, 'same')>0.5); % predicted nucleus mask with shape smoothing
    TP = nnz(BW1 == 1 & BW2 == 1); % true positives
    FP = nnz(BW2 == 0 & BW1 == 1); % false positives
    FN = nnz(BW2 == 1 & BW1 == 0); % false negatives
    F1(rr) = TP/(TP+0.5*FN+0.5*FP);
    IoU(rr) = TP/(TP+FN+FP);
    if isShow == 1
        delete(tt1)
        delete(tt2)
        delete(tt3)

        subplot(s1)
        cla(s1)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(x,2));axis image;axis off;
        set(gca,'Fontsize',15)
        colormap(s1,'gray')
        M = max(x(:));
        caxis([0 round(M)])
        hh = colorbar('Position',[0.395045517502267,0.545086120914231,0.011204481792717,0.285714282994233]);
        xlabel(hh,'[rad]')
        set(hh,'XTick',[0:2:round(M)])
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['INPUT'],['Stain-Free QPM']},'Color',[64 64 64]/255,'Fontsize',18)
    
    
        subplot(s2)
        cla(s2)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(x,2));axis image;axis off;
        pp = bwboundaries(imresize(BW1,2)>0.5);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'-','Linewidth',4,'Color',[0 0.5 0])
        pp = bwboundaries(imresize(BW2,2)>0.5);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'-r','Linewidth',4)
        set(gca,'Fontsize',15)
        colormap(s2,'gray')
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['MERGE'],['In-Silico Stained QPM']},'Color',[0 112 192]/255,'Fontsize',18)
    
        subplot(s3)
        cla(s3)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(BW2,2)>0.5);axis image;axis off;
        BWcell = double(imresize(x>0,2)>0.5);
        pp = bwboundaries(BWcell);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'b--','Linewidth',4)
        colormap(s3,[0 0 0;1 0 0])
        set(gca,'Fontsize',15)
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['TARGET'],['Reprojected Nucleus Mask']},'Color',[255 0 0]/255,'Fontsize',18)
    
    
        subplot(s4)
        cla(s4)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(BW1,2)>0.5);axis image;axis off;
        BWcell = double(imresize(x>0,2)>0.5);
        pp = bwboundaries(BWcell);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'b--','Linewidth',4)
        colormap(s4,[0 0 0;0 0.5 0])
        set(gca,'Fontsize',15)
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['OUTPUT'],['Predicted Nucleus Mask']},'Color',[0 128 0]/255,'Fontsize',18)

        tt1 = annotation('textbox', [0.275 0.20 0.5 0.5], 'String', ['Image = ' num2str(rr)], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        tt2 = annotation('textbox', [0.275 0.10 0.5 0.5], 'String', ['F1 Score = ' num2str(F1(rr),'%.3f')], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        tt3 = annotation('textbox', [0.275 0 0.5 0.5], 'String', ['IoU = ' num2str(IoU(rr),'%.3f')], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        drawnow
    end
end

F1_Validation = F1;
IoU_Validation = IoU;

%% Internal Test Set
D = ['Dataset\Internal Test\Input'];
imds1 = imageDatastore(D);
D = ['Dataset\Internal Test\Target'];
imds2 = pixelLabelDatastore(D,["BG" "N"],[0 255]);
imds2.ReadFcn = @(loc)transformIMG_label(imread(loc));
XX = imds1;
YY = imds2;

if isShow == 1
    figure('units','normalized','outerposition',[0 0 1 1])
    sgtitle('     Internal Test Set - HeLa Cells','Fontsize',36)
    s1 = subplot(2,2,1);
    s2 = subplot(2,2,2);
    s3 = subplot(2,2,3);
    s4 = subplot(2,2,4);
    tt1 = [];
    tt2 = [];
    tt3 = [];
end
F1 = zeros(length(XX.Files),1);
IoU = zeros(length(XX.Files),1);
N = 7;
kernel = ones(N, N) / N^2;
for rr = 1:length(XX.Files)
    rr
    x = readimage(XX,rr);
    y = readimage(YY,rr);
    BW2 = double(y)-1; % target nucleus mask
    z = (predict(net,x));
    BW1 = double(convn(double(diff(z,[],3)>0), kernel, 'same')>0.5); % predicted nucleus mask with shape smoothing
    TP = nnz(BW1 == 1 & BW2 == 1); % true positives
    FP = nnz(BW2 == 0 & BW1 == 1); % false positives
    FN = nnz(BW2 == 1 & BW1 == 0); % false negatives
    F1(rr) = TP/(TP+0.5*FN+0.5*FP);
    IoU(rr) = TP/(TP+FN+FP);
    if isShow == 1
        delete(tt1)
        delete(tt2)
        delete(tt3)

        subplot(s1)
        cla(s1)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(x,2));axis image;axis off;
        set(gca,'Fontsize',15)
        colormap(s1,'gray')
        M = max(x(:));
        caxis([0 round(M)])
        hh = colorbar('Position',[0.395045517502267,0.545086120914231,0.011204481792717,0.285714282994233]);
        xlabel(hh,'[rad]')
        set(hh,'XTick',[0:2:round(M)])
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['INPUT'],['Stain-Free QPM']},'Color',[64 64 64]/255,'Fontsize',18)
    
    
        subplot(s2)
        cla(s2)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(x,2));axis image;axis off;
        pp = bwboundaries(imresize(BW1,2)>0.5);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'-','Linewidth',4,'Color',[0 0.5 0])
        pp = bwboundaries(imresize(BW2,2)>0.5);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'-r','Linewidth',4)
        set(gca,'Fontsize',15)
        colormap(s2,'gray')
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['MERGE'],['In-Silico Stained QPM']},'Color',[0 112 192]/255,'Fontsize',18)
    
        subplot(s3)
        cla(s3)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(BW2,2)>0.5);axis image;axis off;
        BWcell = double(imresize(x>0,2)>0.5);
        pp = bwboundaries(BWcell);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'b--','Linewidth',4)
        colormap(s3,[0 0 0;1 0 0])
        set(gca,'Fontsize',15)
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['TARGET'],['Reprojected Nucleus Mask']},'Color',[255 0 0]/255,'Fontsize',18)
    
    
        subplot(s4)
        cla(s4)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(BW1,2)>0.5);axis image;axis off;
        BWcell = double(imresize(x>0,2)>0.5);
        pp = bwboundaries(BWcell);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'b--','Linewidth',4)
        colormap(s4,[0 0 0;0 0.5 0])
        set(gca,'Fontsize',15)
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['OUTPUT'],['Predicted Nucleus Mask']},'Color',[0 128 0]/255,'Fontsize',18)

        tt1 = annotation('textbox', [0.275 0.20 0.5 0.5], 'String', ['Image = ' num2str(rr)], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        tt2 = annotation('textbox', [0.275 0.10 0.5 0.5], 'String', ['F1 Score = ' num2str(F1(rr),'%.3f')], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        tt3 = annotation('textbox', [0.275 0 0.5 0.5], 'String', ['IoU = ' num2str(IoU(rr),'%.3f')], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        drawnow
    end
end

F1_Internal_Test = F1;
IoU_Internal_Test = IoU;

%% Independent Internal Test Set
D = ['Dataset\Independent Internal Test\Input'];
imds1 = imageDatastore(D);
D = ['Dataset\Independent Internal Test\Target'];
imds2 = pixelLabelDatastore(D,["BG" "N"],[0 255]);
imds2.ReadFcn = @(loc)transformIMG_label(imread(loc));
XX = imds1;
YY = imds2;

if isShow == 1
    figure('units','normalized','outerposition',[0 0 1 1])
    sgtitle('     Independent Internal Test Set - HeLa Cells','Fontsize',36)
    s1 = subplot(2,2,1);
    s2 = subplot(2,2,2);
    s3 = subplot(2,2,3);
    s4 = subplot(2,2,4);
    tt1 = [];
    tt2 = [];
    tt3 = [];
end
F1 = zeros(length(XX.Files),1);
IoU = zeros(length(XX.Files),1);
N = 7;
kernel = ones(N, N) / N^2;
for rr = 1:length(XX.Files)
    rr
    x = readimage(XX,rr);
    y = readimage(YY,rr);
    BW2 = double(y)-1; % target nucleus mask
    z = (predict(net,x));
    BW1 = double(convn(double(diff(z,[],3)>0), kernel, 'same')>0.5); % predicted nucleus mask with shape smoothing
    TP = nnz(BW1 == 1 & BW2 == 1); % true positives
    FP = nnz(BW2 == 0 & BW1 == 1); % false positives
    FN = nnz(BW2 == 1 & BW1 == 0); % false negatives
    F1(rr) = TP/(TP+0.5*FN+0.5*FP);
    IoU(rr) = TP/(TP+FN+FP);
    if isShow == 1
        delete(tt1)
        delete(tt2)
        delete(tt3)

        subplot(s1)
        cla(s1)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(x,2));axis image;axis off;
        set(gca,'Fontsize',15)
        colormap(s1,'gray')
        M = max(x(:));
        caxis([0 round(M)])
        hh = colorbar('Position',[0.395045517502267,0.545086120914231,0.011204481792717,0.285714282994233]);
        xlabel(hh,'[rad]')
        set(hh,'XTick',[0:2:round(M)])
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['INPUT'],['Stain-Free QPM']},'Color',[64 64 64]/255,'Fontsize',18)
    
    
        subplot(s2)
        cla(s2)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(x,2));axis image;axis off;
        pp = bwboundaries(imresize(BW1,2)>0.5);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'-','Linewidth',4,'Color',[0 0.5 0])
        pp = bwboundaries(imresize(BW2,2)>0.5);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'-r','Linewidth',4)
        set(gca,'Fontsize',15)
        colormap(s2,'gray')
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['MERGE'],['In-Silico Stained QPM']},'Color',[0 112 192]/255,'Fontsize',18)
    
        subplot(s3)
        cla(s3)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(BW2,2)>0.5);axis image;axis off;
        BWcell = double(imresize(x>0,2)>0.5);
        pp = bwboundaries(BWcell);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'b--','Linewidth',4)
        colormap(s3,[0 0 0;1 0 0])
        set(gca,'Fontsize',15)
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['TARGET'],['Reprojected Nucleus Mask']},'Color',[255 0 0]/255,'Fontsize',18)
    
    
        subplot(s4)
        cla(s4)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(BW1,2)>0.5);axis image;axis off;
        BWcell = double(imresize(x>0,2)>0.5);
        pp = bwboundaries(BWcell);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'b--','Linewidth',4)
        colormap(s4,[0 0 0;0 0.5 0])
        set(gca,'Fontsize',15)
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['OUTPUT'],['Predicted Nucleus Mask']},'Color',[0 128 0]/255,'Fontsize',18)

        tt1 = annotation('textbox', [0.275 0.20 0.5 0.5], 'String', ['Image = ' num2str(rr)], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        tt2 = annotation('textbox', [0.275 0.10 0.5 0.5], 'String', ['F1 Score = ' num2str(F1(rr),'%.3f')], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        tt3 = annotation('textbox', [0.275 0 0.5 0.5], 'String', ['IoU = ' num2str(IoU(rr),'%.3f')], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        drawnow
    end
end

F1_Independent_Internal_Test = F1;
IoU_Independent_Internal_Test = IoU;

%% Independent External Test Set
D = ['Dataset\Independent External Test\Input'];
imds1 = imageDatastore(D);
D = ['Dataset\Independent External Test\Target'];
imds2 = pixelLabelDatastore(D,["BG" "N"],[0 255]);
imds2.ReadFcn = @(loc)transformIMG_label(imread(loc));
XX = imds1;
YY = imds2;

if isShow == 1
    figure('units','normalized','outerposition',[0 0 1 1])
    sgtitle('     External Internal Test Set - MCF-7 Cells','Fontsize',36)
    s1 = subplot(2,2,1);
    s2 = subplot(2,2,2);
    s3 = subplot(2,2,3);
    s4 = subplot(2,2,4);
    tt1 = [];
    tt2 = [];
    tt3 = [];
end
F1 = zeros(length(XX.Files),1);
IoU = zeros(length(XX.Files),1);
N = 7;
kernel = ones(N, N) / N^2;
for rr = 1:length(XX.Files)
    rr
    x = readimage(XX,rr);
    y = readimage(YY,rr);
    BW2 = double(y)-1; % target nucleus mask
    z = (predict(net,x));
    BW1 = double(convn(double(diff(z,[],3)>0), kernel, 'same')>0.5); % predicted nucleus mask with shape smoothing
    TP = nnz(BW1 == 1 & BW2 == 1); % true positives
    FP = nnz(BW2 == 0 & BW1 == 1); % false positives
    FN = nnz(BW2 == 1 & BW1 == 0); % false negatives
    F1(rr) = TP/(TP+0.5*FN+0.5*FP);
    IoU(rr) = TP/(TP+FN+FP);
    if isShow == 1
        delete(tt1)
        delete(tt2)
        delete(tt3)

        subplot(s1)
        cla(s1)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(x,2));axis image;axis off;
        set(gca,'Fontsize',15)
        colormap(s1,'gray')
        M = max(x(:));
        caxis([0 round(M)])
        hh = colorbar('Position',[0.395045517502267,0.545086120914231,0.011204481792717,0.285714282994233]);
        xlabel(hh,'[rad]')
        set(hh,'XTick',[0:2:round(M)])
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['INPUT'],['Stain-Free QPM']},'Color',[64 64 64]/255,'Fontsize',18)
    
    
        subplot(s2)
        cla(s2)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(x,2));axis image;axis off;
        pp = bwboundaries(imresize(BW1,2)>0.5);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'-','Linewidth',4,'Color',[0 0.5 0])
        pp = bwboundaries(imresize(BW2,2)>0.5);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'-r','Linewidth',4)
        set(gca,'Fontsize',15)
        colormap(s2,'gray')
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['MERGE'],['In-Silico Stained QPM']},'Color',[0 112 192]/255,'Fontsize',18)
    
        subplot(s3)
        cla(s3)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(BW2,2)>0.5);axis image;axis off;
        BWcell = double(imresize(x>0,2)>0.5);
        pp = bwboundaries(BWcell);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'b--','Linewidth',4)
        colormap(s3,[0 0 0;1 0 0])
        set(gca,'Fontsize',15)
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['TARGET'],['Reprojected Nucleus Mask']},'Color',[255 0 0]/255,'Fontsize',18)
    
    
        subplot(s4)
        cla(s4)
        imagesc([-12:0.125:12],[-12:0.125:12],imresize(BW1,2)>0.5);axis image;axis off;
        BWcell = double(imresize(x>0,2)>0.5);
        pp = bwboundaries(BWcell);
        pp = pp{1};
        hold on
        plot((pp(:,2)-96)*0.125,(pp(:,1)-96)*0.125,'b--','Linewidth',4)
        colormap(s4,[0 0 0;0 0.5 0])
        set(gca,'Fontsize',15)
        axis on
        xticks([-12:6:12])
        yticks([-12:6:12])
        xlabel(['x [' char(956) 'm]'])
        ylabel(['y [' char(956) 'm]'])
        title({['OUTPUT'],['Predicted Nucleus Mask']},'Color',[0 128 0]/255,'Fontsize',18)

        tt1 = annotation('textbox', [0.275 0.20 0.5 0.5], 'String', ['Image = ' num2str(rr)], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        tt2 = annotation('textbox', [0.275 0.10 0.5 0.5], 'String', ['F1 Score = ' num2str(F1(rr),'%.3f')], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        tt3 = annotation('textbox', [0.275 0 0.5 0.5], 'String', ['IoU = ' num2str(IoU(rr),'%.3f')], ...
        'FitBoxToText', 'on', 'BackgroundColor', 'none', 'EdgeColor', 'none','Fontsize',27,'HorizontalAlignment','center');

        drawnow
    end
end

F1_Independent_External_Test = F1;
IoU_Independent_External_Test = IoU;

%% Save Metrics

if isequal(name,'trainedNet_paper') == 0
    save(['metrics_' name '.mat'],'F1_Training','IoU_Training','F1_Validation','IoU_Validation','F1_Internal_Test','IoU_Internal_Test','F1_Independent_Internal_Test','IoU_Independent_Internal_Test','F1_Independent_External_Test','IoU_Independent_External_Test')
end