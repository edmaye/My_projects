clc;clear;
%建立符号变量a(发展系数)和b(灰作用量)
syms a b;
c = [a b]';

% 得到原始数列与累加数列
x = [30.5822, 64.4, 93.2];     
x_add = cumsum(x);               

%对累加数列 x_add 做紧邻均值生成
n = length(x)
for i = 2:n
    x_mean(i) = (x_add(i) + x_add(i - 1))/2;  
end
x_mean(1) = [];

%构造数据矩阵 
B = [-x_mean;ones(1,n-1)];
Y = x; Y(1) = []; Y = Y';

%使用最小二乘法计算参数 a(发展系数)和b(灰作用量)
c = inv(B*B')*B*Y; c = c';
a = c(1); b = c(2);

%预测后续数据(累加数列）
predict_add = []; predict_add(1) = x(1);
for i = 2:(n+4)
    predict_add(i) = (x(1)-b/a)/exp(a*(i-1))+ b/a;
end

%对累加数列 predict_add 累减还原,得到预测出的数据
predict = []; predict(1) = x(1);
for i = 2:(n+4)
    predict(i) = predict_add(i) - predict_add(i-1); %得到预测出来的数据
end
disp('预测数据为：'); predict

%绘制曲线图
year = 2010:5:2040;
plot([2010,2015,2020], x,'ro'); hold on;
plot(year, predict, 'b-');
xlabel('年份'); ylabel('60岁以上人口/万人'); legend('实际人口','预测人口'); title('60岁以上人口增长曲线');
for i=1:7
    text(year(i)+0.5,predict(i),num2str(predict(i)))
end

