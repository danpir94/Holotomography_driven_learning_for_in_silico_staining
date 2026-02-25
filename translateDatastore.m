% This function adds a random translation to each image of the training set at each
% iteration to improve generalization

function out = translateDatastore(data)
I = data{1};
C = data{2};

cx=randi(11)-6;
cy=randi(11)-6;
if cx>=0
    augmentedImage=padarray(I,[0 abs(cx) 0],'post','replicate');
    augmentedLabel=padarray(C,[0 abs(cx) 0],'post','replicate');
    augmentedImage=augmentedImage(:,abs(cx)+1:end,:);
    augmentedLabel=augmentedLabel(:,abs(cx)+1:end,:);
elseif cx<0
    augmentedImage=padarray(I,[0 abs(cx) 0],'pre','replicate');
    augmentedLabel=padarray(C,[0 abs(cx) 0],'pre','replicate');
    augmentedImage=augmentedImage(:,1:end-abs(cx),:);
    augmentedLabel=augmentedLabel(:,1:end-abs(cx),:);
end
if cy>=0
    augmentedImage=padarray(I,[abs(cy) 0 0],'post','replicate');
    augmentedLabel=padarray(C,[abs(cy) 0 0],'post','replicate');
    augmentedImage=augmentedImage(abs(cy)+1:end,:,:);
    augmentedLabel=augmentedLabel(abs(cy)+1:end,:,:);
elseif cy<0
    augmentedImage=padarray(I,[abs(cy) 0 0],'pre','replicate');
    augmentedLabel=padarray(C,[abs(cy) 0 0],'pre','replicate');
    augmentedImage=augmentedImage(1:end-abs(cy),:,:);
    augmentedLabel=augmentedLabel(1:end-abs(cy),:,:);
end  

out = {augmentedImage,augmentedLabel};
end