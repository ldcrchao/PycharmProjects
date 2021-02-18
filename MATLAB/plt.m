function  plt(x,y,titlename,legendname,xlabelname,ylabelname)
%% ����
%x=linspace(0,100,100);
%y=x.^2;
%plt(x,y,'��������','y=x^2','x','y')
%unidrnd(N)�������1-N֮�������
%% ��ͼ��,titlename���ַ�������
plot(x,y,'marker','o','linestyle','-.','color','k','linewidth',1.5,...
'markersize',8,'markerindices',1:10:length(y),'markeredgecolor','none','markerfacecolor','none');
%�߱�־,����,��ɫ,��ߴ�(default=6),��־���̶ܳ�(ÿ10�����־1��),��־��Ե��ɫ,��־�����ɫ,�ᱻcolor����
%% ��ɫ��� 
colormap flag%��ֱ��û��
%% ����
%accurcy=100;
%string={titlename,['accurcy^2=',num2str(accurcy),'%'],'y_1=\alpha^{\lambda x}'};%^Ϊ�ϱ꣬��ʾ�ٷֱȵķ�ʽ
title(titlename,'color','k','fontsize',18,'fontweight','bold','fontname','Roman')%�Ӵֻ�����,��ʹ�����±�,'interpreter','none'
%% �����᷶Χ
xmax=max(x)+1;
xmin=min(x)-1;
ymax=max(y)+1;
ymin=min(y)-1;
xlim([xmin xmax]);
ylim([ymin ymax]);
%% �������ǩ
xlabel(xlabelname,'fontname','Roman','fontsize',12,'fontweight','normal','color','k');
ylabel(ylabelname,'fontname','Roman','fontsize',12,'fontweight','normal','color','k');
%xlabel('\bf \it -2\pi \leq x \leq 2\pi')�Ӵ�/б��/ϣ����ĸ
%xlabel('t_{seconds}')
%ylabel('e^t') ���±�
%t=xlabel(xlabelname);t.color='r'���������ʹ��
%% �̶��߱�ǩ
xticks([xmin (xmin+xmax)/2 xmax]);%xticks(0:10:50)Ҳ����
%xticklabels({'���','�е�','�յ�'});
yticks([ymin (ymin+ymax)/2 ymax]);
%yticklabels({'���','�е�','�յ�'});
xticks('auto')%��ΪĬ��ֵ
yticks('auto')
%xticks([]);
%yticks([]);
%% ����������
%axis(xmin,xmax,ymin,ymax),��xlim������ͬ
axis manual
axis fill%ʹ�������������ͼ��
%axis off
axis square%����������ϵ
axis equal%�ȳ��̶�
axis normal%�ָ�Ĭ��
axis tight%�����ݷ�Χֱ����Ϊ���귶Χ��ȥ���հײ���
%% ���ֱ�ע
str=['���','�е�','�յ�'];
text([10;50;100],[100;2500;10000],[str(1:2);str(3:4);str(5:6)])
%% ͼ��
% leg=legend(legendname,'Location','NorthEast','Fontname','Roman','FontSize',16,'NumColumns',1,...
%   'Orientation','horizontal','TextColor','k')%ֱ�����û����
leg=legend(legendname);
leg.FontSize=16;
leg.FontName='Roman';
leg.NumColumns=1;
leg.Location='NorthEast';
leg.Orientation='horizontal';
leg.TextColor='k';
%����P157��North��South��East��West��NorthEast(Ĭ��)��
%title(leg,'My Legend Title')
legend('boxoff')%����ʾ�߿�
%legend('toggle')%�л��ɼ���
%% ������
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
%% �����������
fig=gca;
%fig.Title.String={}
%fig.Title.FontWeight��
%fig.YLabel.String = ''

%λ��
%fig.OuterPosition=[0.1 0.1 0.8 0.8];%Ĭ��[0 0 1 1],������ǩ�ͱ߾�����
%left �� bottom ֵָʾ��ͼ�����½ǵ���߽����½ǵľ���
%width �� height ֵָʾ��߽�ߴ�,��Ⱥ͸߶�
fig.Position=[0.17 0.16 0.7 0.7];%��������ǩ�ͱ߾�

%����ʽ
fig.Color='w';%���������ɫ
%fig.LineWidth=1;%ͳһ����
fig.Box='on';%�Һ��϶�������,on��ʵ��
%fig.BoxStyle='full';%Ĭ��back,ֻ����ά��Ӱ��
fig.Clipping='on';%����������Χ�ڲü�����Ĭ��on

%����
fig.FontName='Roman';
%fig.FontSize=12;%�������嶼����
fig.FontAngle='normal';%italic�ַ���б
fig.LabelFontSizeMultiplier=1.2;%��ǩ�����С����������
fig.TitleFontSizeMultiplier=1.5;%���������С����������
fig.TitleFontWeight='bold';%�����ַ��Ĵ�ϸ
fig.FontUnits='points';%�����С��λ����ѡinches,centimeters,normalized,pixels
fig.FontSmoothing='on';%�ַ�ƽ������

%�̶�
%fig.XTick=[0:0.1:100];%�̶�ֵ
%fig.YTick=[0:0.1:10000];
%fig.XTickLabel={'Jan','Feb','Mar'};%��ǩֵ
%fig.YTickLabel
fig.TickLabelInterpreter='tex';%�̶ȱ�ǩ�Ľ���,��ѡnone,latex
%fig.XTickLabelRotation=45%�̶ȱ�ǩ��ת
%fig.YTickLabelRotation=45
%fig.XMinorTick='off';%�ο̶��ߣ��ƺ�������
%fig.YMinorTick='on';
fig.TickDir='in';%�̶��߷��򣬿�ѡin,out

%���
%fig.XLim=[min max]
%fig.YLim=[min max]
fig.XColor='k';
fig.YColor='k';%��ǩ+����һ���
%fig.XAxis.Color='r';����һ�µ�ֻ�ı�������ɫ
%fig.YAxis.Color='b';%��������ɫ
fig.XAxisLocation='origin';%������λ��Ĭ��bottom,��ѡtop,origin(����ԭ��)
fig.YAxisLocation='origin';%��ѡleft,right
fig.XDir='normal';%��ѡreverse,�����᷽��ı�
fig.YDir='normal';
fig.XScale='linear';%��ѡlog���̶�����ϵ
fig.YScale='linear';%��ѡlog

%������
fig.XGrid='on';%��������
fig.YGrid='on';%������ֱ����grid on����
fig.Layer='top';% �����ߺͿ̶��ߵ�λ��
fig.GridColor='k';%[0.15 0.15 0.15]��ɫ
fig.GridLineStyle='-.';
fig.GridAlpha=0.7;%��͸��,Ĭ��0.15
fig.XMinorGrid='on';%�ο̶���
fig.YMinorGrid='on';
fig.MinorGridLineStyle='--';
fig.MinorGridColor='k';
fig.MinorGridAlpha=0.3;
end
%%
%Ч�������ԣ�':'ð�š�'x'�桢'+'�Ӻš�'<''>'����ָ�����ҵ������Ρ�'^''v'����������
%Ч�������ԣ�'.'�㡢'*'�Ǻš�'s'='square'�����Ρ�'h'='hexagram'�������Ρ�'p'='pentagram'������Ρ�
%'o'ԲȦ��'d'='diamond'��ʯ
%%
%���ʹ��ţ�-ʵ��(Ĭ��)��--���ߡ�-.�㻭�ߡ�:���ߡ�none����
%%
%g(green)��m(Ʒ��ɫmagenta)��b(blue)��c(��ɫcyan)��w(white)��r(red)��k(��ɫ)��y(yellow)
%%
%��ѡ\ite��\alpha��\beta��\delta��\eta��\theta��\Sigma��\sigma������P158