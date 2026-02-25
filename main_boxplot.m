% This script shows boxplots of the F1 score and the IoU metrics about QPMs
% reported in the paper in Fig. 3 for the training,
% validation, internal test, independent internal test, and independent
% external test sets.

%% Load Metrics

close all
clear all
clc

name = 'trainedNet_paper'; % type name of the trained network. If you want to reproduce exactly the same results of the paper, type 'trainedNet_paper'.
load([name '.mat'],'net','info')
load(['metrics_' name '.mat'])
close(info)

figure('units','normalized','outerposition',[0 0 1 1])
plot(info.TrainingHistory.Iteration,movmean(info.TrainingHistory.Loss,10),'-','Linewidth',2)
hold on
plot(info.ValidationHistory.Iteration,info.ValidationHistory.Loss,'.r','Markersize',20)
set(gca,'Fontsize',30)
ylabel('Loss Function')
xlabel('Iteration')
grid on
set(gca,'GridAlpha',1)
xlim([0 2800])
ylim([2.75 4.75])
yticks([2.75:0.5:4.75])
xticks([0:280:2800])
[~,pos]=min(info.ValidationHistory.Loss);
hold on
plot(info.ValidationHistory.Iteration(pos),info.ValidationHistory.Loss(pos),'o','Markersize',12,'Color',[0 0.5 0],'MarkerFaceColor',[0 0.5 0])
legend('Training Loss','Validation Loss','Best Validation')

pp=get(gca,'Position');
set(gca,'Position',[pp(1) pp(2) pp(3) pp(4)*0.9]);

ax1 = gca; 
ax2 = axes('Position', ax1.Position, 'XAxisLocation', 'top', ...
           'YAxisLocation', 'right', 'Color', 'none');
x_upper=[0:10:100];
set(ax2, 'XLim', ax1.XLim, 'XTick', ax1.XTick, ...
         'XTickLabel', num2str(x_upper(:)), ...
         'YColor', 'none'); 
xlabel(ax2, 'Epoch');
set(gca,'Fontsize',30)

F1 = nan*zeros(length(F1_Independent_External_Test),5);
F1(1:length(F1_Training),1) = F1_Training;
F1(1:length(F1_Validation),2) = F1_Validation;
F1(1:length(F1_Internal_Test),3) = F1_Internal_Test;
F1(1:length(F1_Independent_Internal_Test),4) = F1_Independent_Internal_Test;
F1(1:length(F1_Independent_External_Test),5) = F1_Independent_External_Test;
IoU = nan*zeros(length(IoU_Independent_External_Test),5);
IoU(1:length(IoU_Training),1) = IoU_Training;
IoU(1:length(IoU_Validation),2) = IoU_Validation;
IoU(1:length(IoU_Internal_Test),3) = IoU_Internal_Test;
IoU(1:length(IoU_Independent_Internal_Test),4) = IoU_Independent_Internal_Test;
IoU(1:length(IoU_Independent_External_Test),5) = IoU_Independent_External_Test;

%% Show F1 Score
figure('units','normalized','outerposition',[0.25 0.25 0.55 0.6])
p = polyfit([1:5],nanmedian(F1),2);
f = polyval(p,[1:0.1:5]);
ff = plot([1:0.1:5],f,'r-.');
hold on
boxplot(F1,'symbol', '','Widths',0.5);
set(gca,'XtickLabels',{['Training'],['Validation'],['Internal Test'],['Independent \newlineInternal Test'],['Independent \newlineExternal Test']},'TickLabelInterpreter','tex')
set(gca,'Fontsize',24)
grid on
set(gca,'GridAlpha',1)
set(findobj(gca,'type','line'),'linew',2)
ylabel('F1 Score')
ylim([0.4 1])
yticks([0.4:0.1:1])
bb = findobj(gca,'Tag','Box');
colors = [0.8500 0.3250 0.0980;0.9290 0.6940 0.1250;0.2706 0.1961 0.1804;0.3010 0.7450 0.9330;0.4940 0.1840 0.5560];
colors = flipud(colors);
for j = 1:length(bb)
    patch(get(bb(j), 'XData'), get(bb(j), 'YData'), colors(j,:), 'FaceAlpha', 0.25); 
end
set(findobj(gca,'type', 'line', 'Tag', 'Median'),'linew',4)
set(findobj(gca,'type', 'line', 'Tag', 'Box'),'linew',2)
lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(lines, 'Color', [0 0 0]);
lines = findobj(gcf, 'Tag', 'Box');
set(lines, 'Color', [0 0 0]);
h = findobj('LineStyle','--'); set(h, 'LineStyle','-')
set(ff,'Linewidth',3)

%% Show IoU
figure('units','normalized','outerposition',[0.25 0.25 0.55 0.6])
p = polyfit([1:5],nanmedian(IoU),2);
f = polyval(p,[1:0.1:5]);
ff = plot([1:0.1:5],f,'r-.');
hold on
boxplot(IoU,'symbol', '','Widths',0.5);
set(gca,'XtickLabels',{['Training'],['Validation'],['Internal Test'],['Independent \newlineInternal Test'],['Independent \newlineExternal Test']},'TickLabelInterpreter','tex')
set(gca,'Fontsize',24)
grid on
set(gca,'GridAlpha',1)
set(findobj(gca,'type','line'),'linew',2)
ylabel('IoU')
ylim([0.4 1])
yticks([0.4:0.1:1])
bb = findobj(gca,'Tag','Box');
colors = [0.8500 0.3250 0.0980;0.9290 0.6940 0.1250;0.2706 0.1961 0.1804;0.3010 0.7450 0.9330;0.4940 0.1840 0.5560];
colors = flipud(colors);
for j = 1:length(bb)
    patch(get(bb(j), 'XData'), get(bb(j), 'YData'), colors(j,:), 'FaceAlpha', 0.25); 
end
set(findobj(gca,'type', 'line', 'Tag', 'Median'),'linew',4)
set(findobj(gca,'type', 'line', 'Tag', 'Box'),'linew',2)
lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(lines, 'Color', [0 0 0]);
lines = findobj(gcf, 'Tag', 'Box');
set(lines, 'Color', [0 0 0]);
h = findobj('LineStyle','--'); set(h, 'LineStyle','-')
set(ff,'Linewidth',3)
