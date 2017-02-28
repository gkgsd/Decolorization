% Script to call kmeans++ based segmentation code

img = imread(filename);
[pathstr, name, ext] = fileparts(filename);
[height, width, colors] = size(img);

try
    [klabels, numklabels, clabels, numclabels] = DominantColorExtractionMex(img,cdist,minsize);%image, dist, minsize
%     [bmap] = seg2bmap(clabels,width,height);
%     idx = find(bmap>0);
%     timg = img;
%     timg(idx) = 255;
%     imwrite(timg,strcat('/Users/radhakrishnaachanta/rktemp/',name,'_km',ext),'jpg' );
catch
    warning('Problem using function');
end



