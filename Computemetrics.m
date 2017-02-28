%   Distribution code Version 1.0 -- 09/23/2013 by Cewu Lu.
%
%   The Code is created for computing CCPR, CCFR and E-score based on the method described in the following paper 
%   Cewu Lu, Li Xu, Jiaya Jia, "Contrast Preserving Decolorization with Perception-Based Quality Metrics," International Journal of Computer Vision (IJCV), 2014 
%  
%   The code and the algorithm are for non-comercial use only.
thrNum = double(thrNum);
imNum = double(imNum);

CCPR = zeros(5,thrNum);
CCFR = zeros(5,thrNum);
Escore = zeros(5,thrNum);

SCORE = {};
for ii = 1 : imNum
    im   = im2double(imread([path_base 'images/' num2str(ii) '.png']));
%     img1 = im2double(imread([ 'COLOR250\images\',num2str(ii),'_1.png']));
%     img2 = im2double(imread([ 'COLOR250\images\',num2str(ii),'_2.png']));
%     img3 = im2double(imread([ 'COLOR250\images\',num2str(ii),'_3.png']));
%     img4 = im2double(imread([ 'COLOR250\images\',num2str(ii),'_4.png']));
%     img5 = im2double(imread([ 'COLOR250\images\',num2str(ii),'_5.png']));
%     img6 = im2double(imread([ 'COLOR250\images\',num2str(ii),'_6.png']));
    img_mine = im2double(imread([path_results_mine num2str(ii) '.png']));
    img_2012_Lu = im2double(imread([path_base '2012_Lu/' num2str(ii) '.png']));
    img_2013_Song = im2double(imread([path_base '2013_Song/' num2str(ii) '.png']));
    img_2015_Du = im2double(imread([path_base '2015_Du/' num2str(ii) '.png']));
    img_2015_Liu = im2double(imread([path_base '2015_Liu/' num2str(ii) '.png']));


    IMGs(1).img = img_mine;
    IMGs(2).img = img_2012_Lu;
    IMGs(3).img = img_2013_Song;
    IMGs(4).img = img_2015_Du;
    IMGs(5).img = img_2015_Liu;
%     IMGs(1).img = img1;
%     IMGs(2).img = img2;
%     IMGs(3).img = img3;
%     IMGs(4).img = img4;
%     IMGs(5).img = img5;
%     IMGs(6).img = img6;
    scoreSingle = metric(im, IMGs, thrNum);
    
    SCORE{ii} = scoreSingle; 
    
    CCPR = CCPR + 1/imNum*scoreSingle.CCPR;
    CCFR = CCFR + 1/imNum*scoreSingle.CCFR;
    Escore = Escore + 1/imNum*scoreSingle.E_score;
    
    fprintf('the %d th image is processed! \n', ii)
end

% save('results\SCORE.mat', 'SCORE');
% save('results\CCPR.mat', 'CCPR');
% save('results\CCFR.mat', 'CCFR');
% save('results\Escore.mat', 'Escore');

% close all;
% figure;
% CurvePlot(CCPR, 'CCPR');
% figure;
% CurvePlot(CCFR, 'CCFR');
% figure;
% CurvePlot(Escore, 'Escore');
    
 
