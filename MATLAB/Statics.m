function feature = Statics(data)
% load('matlab.mat')
%h = Statics(data) ;
%返回16种统计量
x_peak=max(abs(data)) ;  %1、峰值
x_mean=mean(data)    ; % 均值
x_mean_f=sum(abs(data))/length(data); % 3、平均幅值
x_std =std(data);  % 4、标准差
x_var=var(data);   %5、方差 ，默认除数(N-1)
x_rms=rms(data) ; %6、均方根
x_skew = skewness(data); %7、偏度
x_kurt = kurtosis(data) ;%8、峭度
x_max =max(data) ;%9、最大值
x_min = min(data) ; %10、最小值
x_rmsm=mean(sqrt(abs(data)),2).^2 ;   %11、均方根幅值
x_ydz = x_peak/x_rmsm ; %12、裕度指标
x_bxz = x_rms / x_mean_f;    %13、波形指标
x_mcz = x_peak / x_mean;   %14、脉冲指标
x_fzz = x_peak / x_rms;   % 15、峰值指标
x_qdz = x_kurt / x_rms;  % 16、峭度指标
feature = [x_peak , x_mean , x_mean_f , x_std , x_var , x_rms , x_skew , x_kurt ,...
    x_max,x_min,x_rmsm, x_ydz , x_bxz ,x_mcz , x_fzz , x_qdz] ;

