function  plt(x,y,titlename,legendname,xlabelname,ylabelname)
%% 描述
%x=linspace(0,100,100);
%y=x.^2;
%plt(x,y,'二次曲线','y=x^2','x','y')
%unidrnd(N)随机生成1-N之间的整数
%% 画图用,titlename是字符串变量
plot(x,y,'marker','o','linestyle','-.','color','k','linewidth',1.5,...
'markersize',8,'markerindices',1:10:length(y),'markeredgecolor','none','markerfacecolor','none');
%线标志,线型,颜色,点尺寸(default=6),标志疏密程度(每10个点标志1次),标志边缘颜色,标志填充颜色,会被color覆盖
%% 颜色风格 
colormap flag%对直线没用
%% 标题
%accurcy=100;
%string={titlename,['accurcy^2=',num2str(accurcy),'%'],'y_1=\alpha^{\lambda x}'};%^为上标，显示百分比的方式
title(titlename,'color','k','fontsize',18,'fontweight','bold','fontname','Roman')%加粗或正常,不使用上下标,'interpreter','none'
%% 坐标轴范围
xmax=max(x)+1;
xmin=min(x)-1;
ymax=max(y)+1;
ymin=min(y)-1;
xlim([xmin xmax]);
ylim([ymin ymax]);
%% 坐标轴标签
xlabel(xlabelname,'fontname','Roman','fontsize',12,'fontweight','normal','color','k');
ylabel(ylabelname,'fontname','Roman','fontsize',12,'fontweight','normal','color','k');
%xlabel('\bf \it -2\pi \leq x \leq 2\pi')加粗/斜体/希腊字母
%xlabel('t_{seconds}')
%ylabel('e^t') 上下标
%t=xlabel(xlabelname);t.color='r'，面向对象使用
%% 刻度线标签
xticks([xmin (xmin+xmax)/2 xmax]);%xticks(0:10:50)也可以
%xticklabels({'起点','中点','终点'});
yticks([ymin (ymin+ymax)/2 ymax]);
%yticklabels({'起点','中点','终点'});
xticks('auto')%变为默认值
yticks('auto')
%xticks([]);
%yticks([]);
%% 坐标轴设置
%axis(xmin,xmax,ymin,ymax),与xlim功能相同
axis manual
axis fill%使坐标充满整个绘图区
%axis off
axis square%正方形坐标系
axis equal%等长刻度
axis normal%恢复默认
axis tight%把数据范围直接设为坐标范围，去除空白部分
%% 文字标注
str=['起点','中点','终点'];
text([10;50;100],[100;2500;10000],[str(1:2);str(3:4);str(5:6)])
%% 图例
% leg=legend(legendname,'Location','NorthEast','Fontname','Roman','FontSize',16,'NumColumns',1,...
%   'Orientation','horizontal','TextColor','k')%直接设置会出错
leg=legend(legendname);
leg.FontSize=16;
leg.FontName='Roman';
leg.NumColumns=1;
leg.Location='NorthEast';
leg.Orientation='horizontal';
leg.TextColor='k';
%见书P157；North、South、East、West、NorthEast(默认)等
%title(leg,'My Legend Title')
legend('boxoff')%不显示边框
%legend('toggle')%切换可见性
%% 网格线
%grid on
%y1=sin(x);
%y2=sin(3*x);
%tiledlayout(2,1)
%ax1=nexttile;
%plot(ax1,x,y1)
%ax2=nexttile;
%plot(ax2,x,y2)
%grid(ax1,'xgrid','on','ygrid','off','linewidth',1,...
% 'gridlinestyle','-.','minorgridlinestyle','-','gridcolor','b','minorgridcolor','r','gridalpha','0.8','minorgridalpha','0.5')
%% 面向对象设置
fig=gca;
%fig.Title.String={}
%fig.Title.FontWeight等
%fig.YLabel.String = ''

%位置
%fig.OuterPosition=[0.1 0.1 0.8 0.8];%默认[0 0 1 1],包括标签和边距在内
%left 和 bottom 值指示从图窗左下角到外边界左下角的距离
%width 和 height 值指示外边界尺寸,宽度和高度
fig.Position=[0.17 0.16 0.7 0.7];%不包括标签和边距

%框样式
fig.Color='w';%背景填充颜色
%fig.LineWidth=1;%统一设置
fig.Box='on';%右和上都是虚线,on是实线
%fig.BoxStyle='full';%默认back,只对三维有影响
fig.Clipping='on';%在坐标区范围内裁剪对象，默认on

%字体
fig.FontName='Roman';
%fig.FontSize=12;%所有字体都改了
fig.FontAngle='normal';%italic字符倾斜
fig.LabelFontSizeMultiplier=1.2;%标签字体大小的缩放因子
fig.TitleFontSizeMultiplier=1.5;%标题字体大小的缩放因子
fig.TitleFontWeight='bold';%标题字符的粗细
fig.FontUnits='points';%字体大小单位，可选inches,centimeters,normalized,pixels
fig.FontSmoothing='on';%字符平滑处理

%刻度
%fig.XTick=[0:0.1:100];%刻度值
%fig.YTick=[0:0.1:10000];
%fig.XTickLabel={'Jan','Feb','Mar'};%标签值
%fig.YTickLabel
fig.TickLabelInterpreter='tex';%刻度标签的解释,可选none,latex
%fig.XTickLabelRotation=45%刻度标签旋转
%fig.YTickLabelRotation=45
%fig.XMinorTick='off';%次刻度线，似乎不管用
%fig.YMinorTick='on';
fig.TickDir='in';%刻度线方向，可选in,out

%标尺
%fig.XLim=[min max]
%fig.YLim=[min max]
fig.XColor='k';
fig.YColor='k';%标签+轴线一起的
%fig.XAxis.Color='r';功能一致但只改变轴线颜色
%fig.YAxis.Color='b';%坐标轴颜色
fig.XAxisLocation='origin';%坐标轴位置默认bottom,可选top,origin(穿过原点)
fig.YAxisLocation='origin';%可选left,right
fig.XDir='normal';%可选reverse,坐标轴方向改变
fig.YDir='normal';
fig.XScale='linear';%可选log，刻度坐标系
fig.YScale='linear';%可选log

%网格线
fig.XGrid='on';%竖网格线
fig.YGrid='on';%两个可直接用grid on代替
fig.Layer='top';% 网格线和刻度线的位置
fig.GridColor='k';%[0.15 0.15 0.15]灰色
fig.GridLineStyle='-.';
fig.GridAlpha=0.7;%不透明,默认0.15
fig.XMinorGrid='on';%次刻度线
fig.YMinorGrid='on';
fig.MinorGridLineStyle='--';
fig.MinorGridColor='k';
fig.MinorGridAlpha=0.3;
end
%%
%效果不明显：':'冒号、'x'叉、'+'加号、'<''>'顶点指向左右的三角形、'^''v'正倒三角形
%效果较明显：'.'点、'*'星号、's'='square'正方形、'h'='hexagram'六角星形、'p'='pentagram'五角星形、
%'o'圆圈、'd'='diamond'钻石
%%
%线型代号：-实线(默认)、--虚线、-.点画线、:点线、none无线
%%
%g(green)、m(品红色magenta)、b(blue)、c(灰色cyan)、w(white)、r(red)、k(黑色)、y(yellow)
%%
%可选\ite、\alpha、\beta、\delta、\eta、\theta、\Sigma、\sigma其它见P158