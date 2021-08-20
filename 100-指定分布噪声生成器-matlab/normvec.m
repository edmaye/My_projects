function vec = normvec(n,u,v)
    % 生成均值为0，方差为1的正态分布噪声
    norm_vec = randn(1,n);
    % 对均值和方差进行线性变换
    vec = v * norm_vec + u;
end
