function y=cobb(k)
global alpha
if alpha>0 & alpha<1
   y=k.^alpha;
else
   y=-inf*abs(k);
   disp('invalid capital share')
end