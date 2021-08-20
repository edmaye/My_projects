%% 第二部分
x = 1:5;
h = 1:5;
y1 = conv(x,h)
y2 = myconv_input(x,h)
y3 = myconv_output(x,h)


%% 第三部分
conv_time = [];
input_conv_time = [];
output_conv_time = [];
for i=1:1000
    u=1:i;
    v=1:i;
    % 计算matlab自带函数conv耗时
    tic
    conv(u,v);
    conv_time = [conv_time,toc];
    % 计算输入端算法耗时
    tic
    myconv_input(u,v);
    input_conv_time = [input_conv_time,toc];
    % 计算输出端算法耗时
    tic
    myconv_output(u,v);
    output_conv_time = [output_conv_time,toc];
end
plot(conv_time);
hold on;
plot(input_conv_time);
hold on;
plot(output_conv_time);
xlabel('数据长度')
ylabel('运算耗时/s')
legend('matlab自带卷积函数','输入端卷积函数','输出端卷积函数')

