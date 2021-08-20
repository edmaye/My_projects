clc
clear all

mydir1='image\';  
mydir2='mask\';  
filenames1=dir([mydir1,'*.jpg']);
filenames2=dir([mydir2,'*.jpg']);
for i=1:length(filenames1)
    filename1=[mydir1,filenames1(i).name];
    filename2=[mydir2,filenames1(i).name];
    
    A=imread(filename1);
    B=imread(filename2);  
    r= A(:,:,1);
    g= A(:,:,2);
    b= A(:,:,3);

    rr = double(r);
    gg = double(g);
    ye =rr-0.95*gg;

    yee = imbinarize(ye);
    yee = ~yee;
    yee = bwareaopen(yee,500) ;

    ye_zong = sum(sum(yee));
    b= B(:,:,3);
    bing = im2bw(b,100/255); 
    bing_zong = sum(sum(bing));

    bili = bing_zong/(ye_zong+bing_zong);
    result(i,1)=bili;
end
xlswrite('out.xls', result); 


