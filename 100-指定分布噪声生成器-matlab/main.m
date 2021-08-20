% 生成噪声序列
vec = normvec(10000,5,1);

% 绘制直方图
[N1,bin1]=hist(vec,(min(vec):(max(vec)-min(vec))/100:max(vec))); 
bar(bin1,N1/100);
title('将序列按幅值等分100后统计频率分布')
xlabel('随机变量X取值')
ylabel('随机变量X幅值等分100后的统计频率分布')

% 比较均值和方差是否与预期一致
m = mean(vec);
v = var(vec);
disp(['预期输入  均值：',num2str(5),', 方差:',num2str(1)])
disp(['生成结果  均值：',num2str(m),', 方差:',num2str(v)])