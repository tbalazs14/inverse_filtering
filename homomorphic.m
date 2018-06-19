clc;
clear all;
close all;

xref1=imread('D:\Dokumentumok\2017_18_1\Dipterv1\pic\1973_1.png');
xref2=imread('D:\Dokumentumok\2017_18_1\Dipterv1\pic\1973_2.png');
xref3=imread('D:\Dokumentumok\2017_18_1\Dipterv1\pic\1973_3.png');

% xref1=imread('C:\Users\tbalazs\Dokumentumok\2017_18_2\dipterv2\dipterv1\pic\1973_1.png');
% xref2=imread('C:\Users\tbalazs\Dokumentumok\2017_18_2\dipterv2\dipterv1\pic\1973_2.png');
% xref3=imread('C:\Users\tbalazs\Dokumentumok\2017_18_2\dipterv2\dipterv1\pic\1973_3.png');

xref10 = rgb2gray(xref1);
xref20 = rgb2gray(xref2);
xref30 = rgb2gray(xref3);

Xref1 = abs(fft2(double(xref10))).^2;
Xref2 = abs(fft2(double(xref20))).^2;
Xref3 = abs(fft2(double(xref30))).^2;

Xref = (Xref1+Xref2+Xref3)/3;

x0=imread('D:\Dokumentumok\2017_18_1\Dipterv1\pic\1973_4.png');
%x0=imread('C:\Users\tbalazs\Dokumentumok\2017_18_2\dipterv2\dipterv1\pic\1973_4.png');
x=rgb2gray(x0);

[x_row,x_column] = size(x);
motion_blur_length = 20;
half_motion_blur = motion_blur_length/2;
h=fspecial('motion',motion_blur_length,0);
%h=fspecial('gaussian',motion_blur_length,3);

%h = createBlur(10);


%y = imfilter(x,h);
y = imfilter(x,h,'symmetric');

%todo 
% y=conv2(x,h);
%conv2eredmeny = conv2(x,h);

 %y = y(:,1+motion_blur_length:x_column+motion_blur_length);
y = double(y);

check = y-double(imfilter(x,h));
[m,n] = size(y);

Y = abs(fft2(y)).^2;

H_hom = sqrt(Y./Xref);
h_hom = real((ifft2(H_hom)));
% for j=1:n
%     for i=1:m
%         if h_hom(i,j)<0
%             h_hom(i,j)=0;
%         end
%     end
% end

hom = deconv(y,h_hom,0.1);
%h_hom = ifft2(H_hom);
%uniformly distributed noise
noise = zeros(m,n);
for i=1:m
    for j=1:n
        noise(i,j) = rand()-0.5;
    end
end

%gaussian noise
var = 10;
sqr_var = sqrt(var);
noise2 = zeros(m,n);
for i=1:m
    for j=1:n
        noise2(i,j) = sqr_var* randn();
    end
end

%y = y+noise;
%y = y+noise2;

%y = conv2(x,h);
%h2=zeros(size(x));
%h2(1,1:31)=h;

%referencia kép
%r1=imread('D:\Dokumentumok\2017_18_1\Dipterv1\pic\cpu.jpg');
%r1=imread('D:\Dokumentumok\2017_18_1\Dipterv1\pic\battery.jpg');
%r=rgb2gray(r1);

%y=imfilter(r,h,'replicate');


% 
% ElogImage=0;
% ElogRef=0;
% dist=60;
% 
% for i=1:6
%     for j=1:6
%         temp_image=imagecrop(y,1+(i-1)*dist,1+(j-1)*dist,dist);
%         temp_ref=imagecrop(r,1+(i-1)*dist,1+(j-1)*dist,dist);
%         T=fft2(temp_image);
%         TR=fft2(temp_ref);
%         
%         T=log(abs(T));
%         TR=log(abs(TR));
%         
%         ElogImage=ElogImage+T;
%         ElogRef=ElogRef+TR;
%         
%     end
% end
% 
% for i=1:5
%     for j=1:6
%         temp_image=imagecrop(y,1+(i-1)*dist+dist/2,1+(j-1)*dist,dist);
%         temp_ref=imagecrop(r,1+(i-1)*dist+dist/2,1+(j-1)*dist,dist);
%         
%         T=fft2(temp_image);
%         TR=fft2(temp_ref);
%         
%         T=log(abs(T));
%         TR=log(abs(TR));
%         
%         ElogImage=ElogImage+T;
%         ElogRef=ElogRef+TR;
%     end
% end
% 
% for i=1:6
%     for j=1:5
%         temp_image=imagecrop(y,1+(i-1)*dist,1+(j-1)*dist+dist/2,dist);
%         temp_ref=imagecrop(r,1+(i-1)*dist,1+(j-1)*dist+dist/2,dist);
%         
%         T=fft2(temp_image);
%         TR=fft2(temp_ref);
%         
%         T=log(abs(T));
%         TR=log(abs(TR));
%         
%         ElogImage=ElogImage+T;
%         ElogRef=ElogRef+TR;
%     end
% end
% 
% ElogImage=ElogImage/96;
% ElogRef=ElogRef/96;
% 
% Ref=exp(ElogRef);
% 
% 
% 
% 
%  temp_image=imagecrop(x,1+3*dist,1+4*dist,dist);
%  imshow(uint8(temp_image));
% 
% A=exp(1./Ref.*ElogImage-ElogRef);
% 
% a=ifft2(A);

% J = deconvreg(y, a);



zero_m = zeros(size(y));

%zero padding
y_z = [y zero_m; zero_m zero_m];

%constant padding:
    %[m,n] = size(y);
    % f_r = sum(y(1,:))/n;
    % l_r = sum(y(m,:))/n;
    % f_c = sum(y(:,1))/m;
    % l_c = sum(y(:,n))/m;
    % 
    % c = (f_r+l_r+f_c+l_c)/4;
    % 
    % c_m = zeros(size(y));
    % c_m(:,:) = c;
[m,n] = size(y);
constantR = y(:,n);
constantL = y(:,1);
constantU = y(1,:);
constantB = y(m,:);

constantRM = zeros(m,n);
constantLM = zeros(m,n);
constantUM = zeros(m,n);
constantBM = zeros(m,n);

for i=1:n
constantRM(:,i) = constantR;
constantLM(:,i) = constantL;
end

for i=1:m
constantUM(i,:) = constantU;
constantBM(i,:) = constantU;
end

y_c = [y constantRM; constantBM (constantUM+constantLM)/2];

O = deconv(y, h,0.001);
Z_P = deconv(y_z,h,0.001);
C_P = deconv(y_c,h,0.001);

%reflective padding

horizontal_mirrored = flipud(y);
vertical_mirrored = fliplr(y);
central_mirrored = flipud(vertical_mirrored);

y_r = [y vertical_mirrored; horizontal_mirrored central_mirrored];

hom_r = deconv(y_r,h_hom,0.001);

R_P = deconv(y_r,h,0.001);


figure; imshow(O,[0,255]); title('def');
figure; imshow(Z_P(1:m,1:n),[0,255]); title('zero padding');
figure; imshow(C_P(1:m,1:n),[0,255]); title('constant padding');
figure; imshow(R_P(1:m,1:n),[0,255]); title('reflective padding');
figure; imshow(hom(1:m,1:n),[0,255]); title('homomorf');
figure; imshow(hom_r(1:m,1:n),[0,255]); title('homomorf reflective');



%%homomorphic blind deconvolution
%1/Q E log|Bk | = 1/Q E log|Ik | + log|A|

%Ik: referencia, hibátlan kép
%log | P | = 1/Q E log | Ik | 

% 1/P E log|Bk| - log |P| = log|A|

% |A| kb= exp((E log|Bk|-log|P|)/P)

function x = deconv(y,h,lambda)
Y = fft2(y);
h_pad = zeros(size(y));
[hm,hn] = size(h);
h_pad(1:hm,1:hn) = h;
H = fft2(h_pad);
X = Y.*conj(H)./(lambda+abs(H).^2);
x = uint8(real(ifft2(X)));
end

function cropped = imagecrop (in,x,y,d)

cropped=zeros(d,d);
    for i=1:d
        for j=1:d
            cropped(i,j)=in(x+i-1,y+j-1);
        end
    end
    [r,c]=size(cropped); 
    w=window2(r,c,@hamming); 
    cropped=cropped.*w;
end


function w=window2(N,M,w_func)

wc=window(w_func,N);
wr=window(w_func,M);
[maskr,maskc]=meshgrid(wr,wc);

%maskc=repmat(wc,1,M); Old version
%maskr=repmat(wr',N,1);

w=maskr.*maskc;

end

% function psf = createBlur(c,len)
% nx = rand(size(1:len));
% ny = rand(size(1:len));
% constantX = rand(size(1:len));
% constantY = rand(size(1:len));
% cnx = 2*cumsum(nx).*constantX;
% cny = 2*cumsum(ny).*constantY;
% 
% psf = [cnx;cny];
% psf = round(psf);
% 
% ps = zeros(len);
% for i=1:N-1
%     ps(psf(i,:)
% end
% end

function psf = createBlur(N)
%N=30;
b=0;
nx=rand(size(1:N-1))+b; ny=rand(size(1:N-1))+b;
cnx=cumsum(nx)-N*b/2+1; cny=cumsum(ny)-N*b/2+1;
psf=zeros(N,N);
for i=1:N-1
    psf(round(cnx(i)),round(cny(i)))=psf(round(cnx(i)),round(cny(i)))+1;
end
psf=psf/sum(sum(psf));

%figure(1), mesh(psf)
%figure(2), contour(psf)
%figure(3), imagesc(psf)
end
