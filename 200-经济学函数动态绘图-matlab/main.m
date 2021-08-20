clc;clear all;
param = [0.3,0.05,0.2,0.3]
t1=param(1);    % ρ
t2=param(2);    % n
t3=param(3);    % zita
t4=param(4);    % α

% 窗口设置
figure(1)
axis([0 7 0 1.6])
set(figure(1),'Position',[100,200,1000,400]);

% 画第一条线
kk=0:0.1:7;
h1 = plot(kk,kk.^t4-(t2+t3)*kk);
hold on
h2 = plot(0.8*ones(161,1),0:0.01:1.6)
hold on
h3 = plot(0.5*ones(17,1),0:0.1:1.6,'--r')
hold on
%legend('稳定资本','稳定消费','k=0.5')
flag = 0
F=getframe(gcf);
I=frame2im(F);
[I,map]=rgb2ind(I,256);
imwrite(I,map,'test.gif','gif', 'Loopcount',inf,'DelayTime',0.2);


flag = 0
for k_max=0.51:0.07:2
    kk = [0.5,k_max];
    c = 0.6*kk+0.25;
    h4 = plot(kk,c,'r')
    hold on
    if flag==0
        %legend('稳定资本','稳定消费','k=0.5','鞍点路径')
        flag = 1
    end
    axis([0 7 0 1.6])
    drawnow;
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);
    imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.2);

end
hold on


% 画第二条线
flag = 0
for k_max=0.501:-0.05:0.1
    kk = [0.5,k_max];
    c = -0.5*kk+1.1;
    h5 = plot(kk,c,'b')
    if flag==0
        %legend('非平衡路径1')
        flag = 1
    end
    axis([0 7 0 1.6])
    drawnow;
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);
    imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.2);
end
h5 = quiver(kk(1),c(1),kk(2)-kk(1),c(2)-c(1),'b','MaxHeadSize',1.0,'AutoScaleFactor',0.89,'AutoScale','off');
%legend('稳定资本','稳定消费','k=0.5','鞍点路径','非平衡路径1')
drawnow;

% 第三条线
for k_max=0.51:0.1:3.5
    kk = [0.5,k_max];
    c = -0.1*kk+0.4;
    plot(kk,c,'g')
    axis([0 7 0 1.6])
    drawnow;
    F=getframe(gcf);
    I=frame2im(F);
    [I,map]=rgb2ind(I,256);
    imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.2);
end
h6 = quiver(kk(1),c(1),kk(2)-kk(1),c(2)-c(1),'g','MaxHeadSize',0.1,'AutoScaleFactor',0.89,'AutoScale','off');
legend([h1,h2,h3,h4,h5,h6],'稳定资本','稳定消费','k=0.5','鞍点路径','非平衡路径1','非平衡路径2')
drawnow;
F=getframe(gcf);
I=frame2im(F);
[I,map]=rgb2ind(I,256);
imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.2);
% 

% kk = [0.5,0.1];
% c = -0.5*kk+1.1;
% quiver(kk(1),c(1),kk(2)-kk(1),c(2)-c(1),'MaxHeadSize',1.0,'AutoScaleFactor',0.89,'AutoScale','off');
% legend('稳定资本','稳定消费','k=0.5','鞍点路径','非平衡路径1')
% hold on
% 
% 
% 
% 
% kk = [0.5,4];
% c = -0.1*kk+0.4;
% quiver(kk(1),c(1),kk(2)-kk(1),c(2)-c(1));
% 
% 
% 
% 



% 
% get(gca,'position'); % 此句可查看position默认值
% set(gca,'position',[0.1 0.1 0.8 0.8]); %[ 0 0 1 1]为新设定position值


