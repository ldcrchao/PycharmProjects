function [locs , values] = FindExtremum(x)
%测试数据
% x=[1, 2, 3, 3.1, 5, 7, 8, 9, 9.1, 9.2, 9.4, 9.6, 9.8, 10, 10.2, 10.3, 10.8, 11, 10.8, 10.6, 10.1, 10, 9.5, 9.3, 9,...
%          6, 5.8, 5.7, 5.6, 5.5,...
%          5.3, 5.1, 8, 8.8, 9, 9.5, 10, 10.4, 11, 15, 16, 18, 17.5, 17, 16, 14.4, 14, 13.5, 13, 12, 9.4, 9, 8.8, 7.5, 7]
% [a,b]=FindExtremum(x);
%返回极大值、极小值的横坐标 ； 极大值和极小值
[values_max,locs_1]=findpeaks(x);
[values_min,locs_2]=findpeaks(-x);
plot(x);
hold on
plot(locs_1,values_max,'ro');
hold on 
plot(locs_2,-values_min,'g*');
axis tight
values=[values_max , values_min];
locs = [locs_1,locs_2];
