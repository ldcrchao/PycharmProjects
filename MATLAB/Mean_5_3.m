function [denoise] = Mean_5_3(series,m)
%行向量，列向量均可
%五点三次滤波法
% 简单的滤波处理，主要目的在于消除毛刺和去除趋势项，寻找最大和最小值。
N = length(series);   %数据的长度
a = series;
b = a;
for M = 1:m
    b(1) = (69 * a(1) + 4* a(2) - 6*a(3) + 4*a(3) - a(4)) / 70;
    b(2) = (2 * a(1) + 27 * a(2) + 12 * a(3) - 8*a(3) + 2*a(4)) / 30;
    for j = 3:N-2
        b(j) = (-3*a(j-2) + 12*a(j-1) + 17*a(j) + 12*a(j+1) -3*a(j+2)) /35;
    end
    b(N-1) = (2*a(N-4) - 8 * a(N-3) + 12 * a(N-2) +27*a(N-1)+ 2 * a(N)) / 35;
    b(N) = (-a(N-4) +4*a(N-3) - 6* a(N-2) + 4 * a(N-1) + 69*a(N) ) / 70;
    a = b;
end
denoise=a;
end

 
