function [y] = myconv_input(x, h)
    % 卷积算法的输入端形式
    y = linspace(0, 0, length(x)+length(h)-1);
    for i=1:length(x)
        for j=1:length(h)
            y(i+j-1) = y(i+j-1) + x(i) * h(j);
        end
    end
end
