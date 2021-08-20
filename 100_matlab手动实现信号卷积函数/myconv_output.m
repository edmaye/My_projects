function [y] = myconv_output(x, h)
    % 卷积算法的输出端版本
    y = linspace(0, 0, length(x)+length(h)-1);
    for i=1:length(y)
       for j=1:length(x)
            if i-j+1>0 && i-j+1 <length(h)+1
               y(i) = y(i) + x(j) * h(i-j+1);  
            end
       end
    end
end