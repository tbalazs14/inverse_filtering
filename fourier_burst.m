close all;
clear all;

N = 6;
p = 1;
g = 6;

M=10;

x10 = imread('D:\Dokumentumok\2017_18_1\Dipterv1\burst\01.jpg');
x20 = imread('D:\Dokumentumok\2017_18_1\Dipterv1\burst\02.jpg');
x30 = imread('D:\Dokumentumok\2017_18_1\Dipterv1\burst\05.jpg');
x40 = imread('D:\Dokumentumok\2017_18_1\Dipterv1\burst\07.jpg');
x50 = imread('D:\Dokumentumok\2017_18_1\Dipterv1\burst\09.jpg');
x60 = imread('D:\Dokumentumok\2017_18_1\Dipterv1\burst\11.jpg');

xref = imread('D:\Dokumentumok\2017_18_1\Dipterv1\pic\glasses.jpg');

 x(:,:,:,1)=imresize(x10,0.25); %még aluláteresztõ szûr is
 x(:,:,:,2)=imresize(x20,0.25);
 x(:,:,:,3)=imresize(x30,0.25);
 x(:,:,:,4)=imresize(x40,0.25);
 x(:,:,:,5)=imresize(x50,0.25);
 x(:,:,:,6)=imresize(x60,0.25);
 
 for i=1:M
    angle(i) = 360*rand();
    length(i) = 20+40*rand();

 end
 
 [aa,bb,cc,dd] = size(x);
 ref = zeros(aa,bb,cc);
 ref(aa/2,bb/2,1) = 255;
 ref(aa/2,bb/2,2) = 255;
 ref(aa/2,bb/2,3) = 255;
 
 h1 = fspecial('motion',length(1),angle(1));
 h2 = fspecial('motion',length(2),angle(2));
 h3 = fspecial('motion',length(3),angle(3));
 h4 = fspecial('motion',length(4),angle(4));
 h5 = fspecial('motion',length(5),angle(5));
 h6 = fspecial('motion',length(6),angle(6));
 
 hh = zeros(11,5);
 hh(4,1) = 1;
 hh(5,2) = 1;
 hh(6,3) = 1;
 hh(7,4) = 1;
 hh(8,5) = 1;
 hh = hh/sum(sum(hh));
 
 
% y1 = imfilter(xref,h1,'symmetric');
% y2 = imfilter(xref,h2,'symmetric');
% y3 = imfilter(xref,h3,'symmetric');
% y4 = imfilter(xref,h4,'symmetric');
% y5 = imfilter(xref,h5,'symmetric');
% y6 = imfilter(xref,h6,'symmetric');

 
y1 = imfilter(x(:,:,:,1),h1,'symmetric');
y2 = imfilter(x(:,:,:,1),h2,'symmetric');
y3 = imfilter(x(:,:,:,1),h3,'symmetric');
y4 = imfilter(x(:,:,:,1),h4,'symmetric');
y5 = imfilter(x(:,:,:,1),h5,'symmetric');
y6 = imfilter(x(:,:,:,1),h6,'symmetric');

y7 = imfilter(ref,h1,'symmetric');

x(:,:,:,1) = y1;
x(:,:,:,2) = y2;
x(:,:,:,3) = y3;
x(:,:,:,4) = y4;
x(:,:,:,5) = y5;
x(:,:,:,6) = y6;

figure(1);
imshow(y1);

figure(2);
imshow(y2);

figure(3);
imshow(y3);

figure(4);
imshow(y4);

figure(5);
imshow(y5);

figure(6);
imshow(y6);


% angle(1) = 2*pi*rand();
% angle(2) = 2*pi*rand();
% angle(3) = 2*pi*rand();
% angle(4) = 2*pi*rand();
% angle(5) = 2*pi*rand();
% 
% h1 = fspecial('motion',2*M,angle(1));
% h2 = fspecial('motion',2*M,angle(2));
% h3 = fspecial('motion',2*M,angle(3));
% h4 = fspecial('motion',2*M,angle(4));
% h5 = fspecial('motion',2*M,angle(5));
 
 for i = 1:N
     xr(:,:,i) = x(:,:,1,i);
     xg(:,:,i) = x(:,:,2,i);
     xb(:,:,i) = x(:,:,3,i);
     
 end
 
 for i=1:N
     Fxr(:,:,i) = fft2(xr(:,:,i));
     Fxg(:,:,i) = fft2(xg(:,:,i));
     Fxb(:,:,i) = fft2(xb(:,:,i));
 end
 for i=1:N
     w(:,:,i) = 1/3*(abs(Fxr(:,:,i))+abs(Fxg(:,:,i))+abs(Fxb(:,:,i)));
     w(:,:,i) = imgaussfilt(w(:,:,i),g);
 end

[m,n,s] = size(xb);

for i=1:3
    Up(:,:,i) = zeros(m,n);
end
for i=1:N
    Up(:,:,1)=Up(:,:,1)+ w(:,:,i).*Fxr(:,:,i);
    Up(:,:,2)=Up(:,:,2)+ w(:,:,i).*Fxg(:,:,i);
    Up(:,:,3)=Up(:,:,3)+ w(:,:,i).*Fxb(:,:,i);
end
ws = zeros(m,n);
for i=1:N
    ws = ws+w(:,:,i);
end
 
 upr = ifft2(Up(:,:,1)./ws);
 upg = ifft2(Up(:,:,2)./ws);
 upb = ifft2(Up(:,:,3)./ws);

UP=zeros(m,n,3); 
UP(:,:,1)=real(upr);
UP(:,:,2)=real(upg);
UP(:,:,3)=real(upb);

figure(7);
imshow(uint8(UP));
 
function [ theta,rho ] = ransac( pts,iterNum,thDist,thInlrRatio )
%RANSAC Use RANdom SAmple Consensus to fit a line
%	RESCOEF = RANSAC(PTS,ITERNUM,THDIST,THINLRRATIO) PTS is 2*n matrix including 
%	n points, ITERNUM is the number of iteration, THDIST is the inlier 
%	distance threshold and ROUND(THINLRRATIO*SIZE(PTS,2)) is the inlier number threshold. The final 
%	fitted line is RHO = sin(THETA)*x+cos(THETA)*y.
%	Yan Ke @ THUEE, xjed09@gmail.com

sampleNum = 2;
ptNum = size(pts,2);
thInlr = round(thInlrRatio*ptNum);
inlrNum = zeros(1,iterNum);
theta1 = zeros(1,iterNum);
rho1 = zeros(1,iterNum);

for p = 1:iterNum
	% 1. fit using 2 random points
	sampleIdx = randIndex(ptNum,sampleNum);
	ptSample = pts(:,sampleIdx);
	d = ptSample(:,2)-ptSample(:,1);
	d = d/norm(d); % direction vector of the line
	
	% 2. count the inliers, if more than thInlr, refit; else iterate
	n = [-d(2),d(1)]; % unit normal vector of the line
	dist1 = n*(pts-repmat(ptSample(:,1),1,ptNum));
	inlier1 = find(abs(dist1) < thDist);
	inlrNum(p) = length(inlier1);
	if length(inlier1) < thInlr, continue; end
	ev = princomp(pts(:,inlier1)');
	d1 = ev(:,1);
	theta1(p) = -atan2(d1(2),d1(1)); % save the coefs
	rho1(p) = [-d1(2),d1(1)]*mean(pts(:,inlier1),2);
end

% 3. choose the coef with the most inliers
[~,idx] = max(inlrNum);
theta = theta1(idx);
rho = rho1(idx);

end
 