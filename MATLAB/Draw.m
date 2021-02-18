function [fig , ax ] = Draw(x,y,DisplayNames)
%% 参数说明
%给定参数x,y一般至少多组数据,其中y是结构体参数；DisplayNames用于设定图例的名称
% 返回图窗、坐标区、图例对象
%% 例子
% x1= (-1:0.2:2) ; 
% y4 = exp(x1) ;
% y5 =exp(x1.*x1) ;
% y6 = {y4,y5};
%DisplayNames ={ 'y=exp(x)','y=exp(x^2)'}; %图例
fig = figure ; 
ax = axes(fig) ;
Markers= {'diamond ','square'} ;
Marker_Colors  ={ [252, 141, 89] / 255 ,[145, 191, 219] / 255};
for i = 1:length(y)
     L = plot(ax ,x ,y{i}) ;
     L.Marker = Markers{i} ;
     L.MarkerFaceColor = Marker_Colors{i} ;
     L.MarkerEdgeColor = 'k' ;
     L.MarkerSize = 4 ;
     L.LineWidth = 1;
     L.DisplayName = DisplayNames{i} ;
    hold on 
end
leg = legend(ax) ;%加上此语句图窗会显示图例对象，否则画出来看不到
end






