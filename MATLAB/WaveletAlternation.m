function Feature = WaveletAlternation(data)
%小波包变换提取节点范数
%data = csvread("C:\Users\chenbei\Desktop\钢\D1.csv");
%cfs = wpcoef(wpt,[3 0]); 读取小波系数
%wpt = wpdec(data,level,'dename')
%wpt = wpdec(data(:,1),3,'db4');
%plot(wpt)画出树
%[wp,x] = wpfun('db2',7); wpfun可以查看每层的图形
%Feature = WaveletAlternation(data)
[~ , column ]  = size(data) ;%原始序列是按列存放的，如1300*3
Feature = zeros(column,8) ; 
for i = 1 : column
     temp = data(:,i) ; %循环取出每列
     level =3;
     wpt = wpdec(temp,level,'db3');
     %读取第3层每个节点的系数，共8个
     coeff_3_0 = wpcoef(wpt,[3,0]);
     coeff_3_1 = wpcoef(wpt,[3,1]);
     coeff_3_2 = wpcoef(wpt,[3,2]);
     coeff_3_3 = wpcoef(wpt,[3,3]);
     coeff_3_4 = wpcoef(wpt,[3,4]);
     coeff_3_5 = wpcoef(wpt,[3,5]);
     coeff_3_6 = wpcoef(wpt,[3,6]);
     coeff_3_7 = wpcoef(wpt,[3,7]);
     %求每层的矩阵范数
     norm_3_0 = norm(coeff_3_0) ;
     norm_3_1 = norm(coeff_3_1) ;
     norm_3_2 = norm(coeff_3_2) ;
     norm_3_3 = norm(coeff_3_3) ;
     norm_3_4 = norm(coeff_3_4) ;
     norm_3_5 = norm(coeff_3_5) ;
     norm_3_6 = norm(coeff_3_6) ;
     norm_3_7 = norm(coeff_3_7) ;
     %合并特征向量
     Feature(i,:) = [norm_3_0,norm_3_1,norm_3_2,norm_3_3,norm_3_4,norm_3_5,norm_3_6,norm_3_7 ];  
          %绘图
%      figure
%      plot(coeff_3_0)
%      figure
%      plot(coeff_3_1)
%      figure
%      plot(coeff_3_2)
%      figure
%      plot(coeff_3_3)
%      figure
%      plot(coeff_3_4)
%      figure
%      plot(coeff_3_5)
%      figure
%      plot(coeff_3_6)
%      figure
%      plot(coeff_3_7)
end

end

