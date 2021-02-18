function adjust_fig(fig , ax  ,label)
%% 参数说明
% 输入图窗、坐标区、x、y轴标签名字label，结构体形式
% label ={'x','y'} ;
%% 设置图窗属性
%fig = figure
fig.Units = 'centimeters' ;%调整默认单位
fig.Position(3:4) = [7 , 5.25] ;%设置图窗的宽度和高度
fig.Color = [1  , 1 , 1]; %默认是灰色，变成白色
%% 设置坐标区属性
%% 常规属性
%axe = axes(fig) ; % 图窗的坐标区对象，此语句执行后就会出现坐标区
ax.Units = 'centimeters' ;
ax.LineWidth = 1 ;
ax.FontName = 'Times New Roman' ; 
ax.FontSize =10 ;
ax.TickLabelInterpreter = 'latex' ; %有latex、tex、none三种属性
% xlim and ylim
%ax.XLim = [0,10] ;
%ax.YLim = [0,10] ;
% xlabel and ylabel
ax.XLabel.String = label{1} ; % 横坐标标签
ax.XLabel.Units = 'normalized';
ax.XLabel.Interpreter = 'latex' ; 
ax.XLabel.Position(1:2) = [0.95  , 0] ; % 以坐标轴的长宽作为参考
% 首先必须将最大值规划到0~1的范围内，所求系数 x*max =0.95(规定的到左或者下方的距离)
ax.YLabel.String = label{2};
ax.YLabel.Units = 'normalized';
ax.YLabel.Interpreter = 'latex' ; 
ax.YLabel.Position(1:2) = [ 0.1, 0.95 ] ; 
% xminortick and yminortick
ax.XMinorTick = 'on' ; % 打开次刻度尺
ax.YMinorTick = 'on' ;
ax.TickLength(1) = 0.02 ; % 设置刻度线的长度
axis tight
%axis off
% legend
xticks([]);
yticks([]);
leg = legend(ax); % 坐标区的图例对象，此语句执行后就会出现图例区
leg.Interpreter = 'latex' ;
%leg.String = 'legend';
leg.FontSize = 10 ;
leg.Box = 'off' ; 
leg.Location =  'NorthEast' ;  %'best'
end






