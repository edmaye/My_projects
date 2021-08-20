function res=select(data,row,col)
    root = data.displacements;
    n = size(root,2);
    res = [];
    for i=1:n
        v = root(i).plot_u_cur_formatted(row,col);
        res(i,1) = v;
    end
end
