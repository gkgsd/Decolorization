% Script to call kmeans++ based segmentation code

close all;
filename = '/Users/bjin/Documents/EPFL_IVRG/project_decolorization/dataset/color2gray/6.png';
img = imread(filename);
[pathstr, name, ext] = fileparts(filename);
[height, width, colors] = size(img);

try
    [klabels, numklabels, clabels, numclabels] = DominantColorExtractionMex(img,15.0,40);%image, dist, minsize
    [bmap] = seg2bmap(clabels,width,height);
    idx = find(bmap>0);
    timg = img;
    timg(idx) = 255;
%     imwrite(timg,strcat('/Users/radhakrishnaachanta/rktemp/',name,'_km',ext),'jpg' );
catch
    warning('Problem using function');
end

imshow(klabels,[]);