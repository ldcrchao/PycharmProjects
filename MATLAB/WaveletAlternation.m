function Feature = WaveletAlternation(data)
%С�����任��ȡ�ڵ㷶��
%data = csvread("C:\Users\chenbei\Desktop\��\D1.csv");
%cfs = wpcoef(wpt,[3 0]); ��ȡС��ϵ��
%wpt = wpdec(data,level,'dename')
%wpt = wpdec(data(:,1),3,'db4');
%plot(wpt)������
%[wp,x] = wpfun('db2',7); wpfun���Բ鿴ÿ���ͼ��
%Feature = WaveletAlternation(data)
[~ , column ]  = size(data) ;%ԭʼ�����ǰ��д�ŵģ���1300*3
Feature = zeros(column,8) ; 
for i = 1 : column
     temp = data(:,i) ; %ѭ��ȡ��ÿ��
     level =3;
     wpt = wpdec(temp,level,'db3');
     %��ȡ��3��ÿ���ڵ��ϵ������8��
     coeff_3_0 = wpcoef(wpt,[3,0]);
     coeff_3_1 = wpcoef(wpt,[3,1]);
     coeff_3_2 = wpcoef(wpt,[3,2]);
     coeff_3_3 = wpcoef(wpt,[3,3]);
     coeff_3_4 = wpcoef(wpt,[3,4]);
     coeff_3_5 = wpcoef(wpt,[3,5]);
     coeff_3_6 = wpcoef(wpt,[3,6]);
     coeff_3_7 = wpcoef(wpt,[3,7]);
     %��ÿ��ľ�����
     norm_3_0 = norm(coeff_3_0) ;
     norm_3_1 = norm(coeff_3_1) ;
     norm_3_2 = norm(coeff_3_2) ;
     norm_3_3 = norm(coeff_3_3) ;
     norm_3_4 = norm(coeff_3_4) ;
     norm_3_5 = norm(coeff_3_5) ;
     norm_3_6 = norm(coeff_3_6) ;
     norm_3_7 = norm(coeff_3_7) ;
     %�ϲ���������
     Feature(i,:) = [norm_3_0,norm_3_1,norm_3_2,norm_3_3,norm_3_4,norm_3_5,norm_3_6,norm_3_7 ];  
          %��ͼ
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

