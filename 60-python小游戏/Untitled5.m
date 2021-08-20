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
plot(kk,kk.^t4-(t2+t3)*kk);
hold on
plot(0.8*ones(161,1),0:0.01:1.6)
hold on
plot(0.5*ones(17,1),0:0.1:1.6,'--r')
hold on
flag = 0
for k_max=0.51:0.01:2
    kk = [0.5,k_max];
    c = 0.6*kk+0.25;
    plot(kk,c)
    if flag==0
        legend('稳定资本','稳定消费','k=0.5','鞍点路径')
        flag = 1
    end
    axis([0 7 0 1.6])
    drawnow;
end
hold on


% 画第二条线
kk=0:0.1:7;
plot(kk,kk.^t4-(t2+t3)*kk);
hold on
plot(0.8*ones(161,1),0:0.01:1.6)
hold on
plot(0.5*ones(17,1),0:0.1:1.6,'--r')
kk = [0.5,2];
c = 0.6*kk+0.25;
plot(kk,c)
axis([0 7 0 1.6])
drawnow;
hold on


legend('稳定资本','稳定消费','k=0.5','鞍点路径','非平衡路径1')
for k_max=0.51:0.01:7
    kk = [0.5,k_max];
    c = -0.5*kk+1.1;
    quiver(kk(1),c(1),kk(2)-kk(1),c(2)-c(1),'MaxHeadSize',1.0,'AutoScaleFactor',0.89,'AutoScale','off');
    axis([0 7 0 1.6])
    drawnow;
end



x = [0.5,0.1];
c = -0.5*kk+1.1;
quiver(kk(1),c(1),kk(2)-kk(1),c(2)-c(1),'MaxHeadSize',1.0,'AutoScaleFactor',0.89,'AutoScale','off');
legend('稳定资本','稳定消费','k=0.5','鞍点路径','非平衡路径1')
hold on




kk = [0.5,4];
c = -0.1*kk+0.4;
quiver(kk(1),c(1),kk(2)-kk(1),c(2)-c(1));







% 
% get(gca,'position'); % 此句可查看position默认值
% set(gca,'position',[0.1 0.1 0.8 0.8]); %[ 0 0 1 1]为新设定position值


