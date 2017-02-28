function scoreSingle = metric(im, IMGs, thrNum)
%   Input Paras: 
%   @im    : Input color image ranging [0,1] 
%   @IMGs    : Input grayscale image ranging [0,1] achieved by all methods 
%   @thrNum  : number of threshold we considered

%   Outputs: 
%   @scoreSingle : score of CCPR, CCFR and E_score in scoreSingle.CCPR, scoreSingle.CCFR and
%   scoreSingle.E_score respectively
%

    Len = length(IMGs);
    
    scoreSingle.CCPR = zeros(Len, thrNum);
    scoreSingle.CCFR = zeros(Len, thrNum);
    scoreSingle.E_score = zeros(Len, thrNum);

    [colorContr_CCPR, colorContr_CCFR, t1, t2] = color_init(im);
    
    for jj = 1 : Len
        
        imgt = IMGs(jj).img;
        [grayContr_CCPR, grayContr_CCFR] = gray_inti(imgt, t1, t2);
        
        for thr = 1 : thrNum

            ccpr = CCPR_fun_fast(colorContr_CCPR, grayContr_CCPR, thr);
            ccfr = CCFR_fun_fast(colorContr_CCFR, grayContr_CCFR, thr);
            E_score = 2*ccpr*ccfr/(ccpr+ccfr);

            scoreSingle.CCPR(jj,thr) = ccpr;
            scoreSingle.CCFR(jj,thr)= ccfr;
            scoreSingle.E_score(jj,thr) = E_score ;
        end
    end
end


function [colorContr_CCPR, colorContr_CCFR, t1, t2] = color_init(im)
 

%% CCPR
	rand('state',0);
	Len =  (size(im,1)*size(im,2));
	t1 = zeros(10*Len,1);
	t2 = zeros(10*Len,1);

	for ii = 1 : 10
		t1(Len*(ii-1)+1:Len*ii) = randperm(Len);
		t2(Len*(ii-1)+1:Len*ii) = randperm(Len);
	end

	imLab = applycform(im,makecform('srgb2lab'));
	L = imLab(:,:,1); A = imLab(:,:,2); B = imLab(:,:,3);
	imV = [L(:),A(:),B(:)];
	colorContr_CCPR = sqrt(sum((imV(t1,:) - imV(t2,:)).^2,2));
    
%% CCFR
    dLx = abs(imfilter(imLab(:,:,1),[1,-1])).^2;
    dAx = abs(imfilter(imLab(:,:,2),[1,-1])).^2;
    dBx = abs(imfilter(imLab(:,:,3),[1,-1])).^2;
    dCx = sqrt(dLx + dAx + dBx); 

    dLy = abs(imfilter(imLab(:,:,1),[1;-1])).^2;
    dAy = abs(imfilter(imLab(:,:,2),[1;-1])).^2;
    dBy = abs(imfilter(imLab(:,:,3),[1;-1])).^2;
    dCy = sqrt(dLy + dAy + dBy); 

    colorContr_CCFR = [dCx(:); dCy(:)]; 

end

function  [grayContr_CCPR, grayContr_CCFR] = gray_inti(img, t1, t2)

    img = repmat(img(:,:,1),[1,1,3]);
    imgLab_Gray = applycform(img,makecform('srgb2lab'));
    imgL = imgLab_Gray(:,:,1);

    %% CCPR
    imgV = imgL(:);
    grayContr_CCPR = abs(imgV(t1) - imgV(t2));
    %% CCFR
    dGx = abs(imfilter(imgL,[1,-1]));
    dGy = abs(imfilter(imgL,[1;-1]));
    grayContr_CCFR = [dGx(:) ; dGy(:)]; 
end


function ccpr = CCPR_fun_fast(colorContr_CCPR, grayContr_CCPR, thr)
    color_local = find(colorContr_CCPR > thr);
    gray_local = find(grayContr_CCPR(color_local) > thr);
    ccpr = length(gray_local)/(length(color_local) + isempty(color_local));
end


function ccfr = CCFR_fun_fast(colorContr_CCFR, grayContr_CCFR, thr)
    gray_locate = find(grayContr_CCFR > thr);
    color_locate = find(colorContr_CCFR(gray_locate) < thr);
    ccfr = 1-length(color_locate)/(length(gray_locate) + isempty(gray_locate));
end


