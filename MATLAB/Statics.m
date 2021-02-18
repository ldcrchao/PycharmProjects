function feature = Statics(data)
% load('matlab.mat')
%h = Statics(data) ;
%����16��ͳ����
x_peak=max(abs(data)) ;  %1����ֵ
x_mean=mean(data)    ; % ��ֵ
x_mean_f=sum(abs(data))/length(data); % 3��ƽ����ֵ
x_std =std(data);  % 4����׼��
x_var=var(data);   %5������ ��Ĭ�ϳ���(N-1)
x_rms=rms(data) ; %6��������
x_skew = skewness(data); %7��ƫ��
x_kurt = kurtosis(data) ;%8���Ͷ�
x_max =max(data) ;%9�����ֵ
x_min = min(data) ; %10����Сֵ
x_rmsm=mean(sqrt(abs(data)),2).^2 ;   %11����������ֵ
x_ydz = x_peak/x_rmsm ; %12��ԣ��ָ��
x_bxz = x_rms / x_mean_f;    %13������ָ��
x_mcz = x_peak / x_mean;   %14������ָ��
x_fzz = x_peak / x_rms;   % 15����ֵָ��
x_qdz = x_kurt / x_rms;  % 16���Ͷ�ָ��
feature = [x_peak , x_mean , x_mean_f , x_std , x_var , x_rms , x_skew , x_kurt ,...
    x_max,x_min,x_rmsm, x_ydz , x_bxz ,x_mcz , x_fzz , x_qdz] ;

