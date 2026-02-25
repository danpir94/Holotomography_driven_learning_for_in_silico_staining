% This function converts the nucleus binary mask into the uint8 format.

function y=transformIMG_label(x)
y=uint8((x==255)*255);
