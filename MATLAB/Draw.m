function [fig , ax ] = Draw(x,y,DisplayNames)
%% ����˵��
%��������x,yһ�����ٶ�������,����y�ǽṹ�������DisplayNames�����趨ͼ��������
% ����ͼ������������ͼ������
%% ����
% x1= (-1:0.2:2) ; 
% y4 = exp(x1) ;
% y5 =exp(x1.*x1) ;
% y6 = {y4,y5};
%DisplayNames ={ 'y=exp(x)','y=exp(x^2)'}; %ͼ��
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
leg = legend(ax) ;%���ϴ����ͼ������ʾͼ�����󣬷��򻭳���������
end






