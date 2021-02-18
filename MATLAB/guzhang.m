clear all;
close all;
clc;
vnew=load('F:\小波分析理论轴承数据\滚动体\225.mat');
fs=12000;
N=3000;
signal=vnew.X225_DE_time;
xdata=signal(1:N);
xdata=(xdata-mean(xdata))/std(xdata,1);
plot(1:N,xdata);
xlabel('时间 t/n');
ylabel('电压 V/v');

%db10小波进行4层分解
%一维小波分解
[c,l] = wavedec(xdata,4,'db10');

a4=wrcoef('a',c,l,'db10',4);
%重构第1～4层细节信号
d4 = wrcoef('d',c,l,'db10',4);
d3 = wrcoef('d',c,l,'db10',3);
d2 = wrcoef('d',c,l,'db10',2);
d1 = wrcoef('d',c,l,'db10',1);

%显示细节信号
figure(2)
subplot(4,1,1);
plot(d4,'LineWidth',2);
ylabel('d4');
subplot(4,1,2);
plot(d3,'LineWidth',2);
ylabel('d3');
subplot(4,1,3);
plot(d2,'LineWidth',2);
ylabel('d2');
subplot(4,1,4);
plot(d1,'LineWidth',2);
ylabel('d1');
xlabel('时间 t/s');

%第1层细节信号的包络谱
d=d1+d2;
y=hilbert(d1);
ydata=abs(y);
y=y-mean(y);
nfft=1024;
p=abs(fft(ydata,nfft));
figure(3);
plot((0:nfft/2-1)/nfft*fs,p(1:nfft/2));
xlabel('频率 f/Hz');
ylabel('功率谱 P/W');

