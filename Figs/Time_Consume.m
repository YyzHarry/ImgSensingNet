%  ImgSensingNet
%  Experiment Results for Tradeoff betweem Total Time and Comsume
%
%  分别画两个图比较:  Total Time - x, BatteryLife - y
%                    BatteryLife - x, Total Time - y 
%

load time_with_sensor.mat time_with_sensor
load time_without_sensor.mat time_without_sensor

%--------------------------------------Total Time Domain-------------------------------------%
%
time_with_sensor(:,81) = 0;
time_without_sensor(:,81) = 0;
time = 0:15:1200;

plot(time,time_with_sensor,'-s',...
                    'LineWidth',2,...
                    'Color',[0.4 0.7 0.2],...
                    'MarkerSize',6);

hold on
plot(time,time_without_sensor,'-v',...
                   'LineWidth',2,...
                   'Color',[0.9 0.3 0.1],...
                   'MarkerSize',6);
%
% axis([0 1 0 1]),
set(gca,'xgrid','on'), set(gca,'ygrid','on')
grid on
xlabel('Total Time (s)'), ylabel('Normalized Battery Consumption (%)');

%
%---------------------------------------Coverage Domain-------------------------------------%
%
figure
cover_load = [2398;
              1687;
              1353;
              1104;
              585
             ];

b = bar(diag(cover_load),0.5,'stack');
color = [[1 0 0.2];[0.9 0.6 0];[1 1 0];[0 1 0.4];[0 0.8 1]];
% [0.9 0.4 0.8]; [0.6 0 1]
for i = 1:5
    set(b(i),'FaceColor',color(i,:));
end

grid on;
set(gca,'XTickLabel',{'w/out sensor','w/ sensor, t=0','w/ sensor, t=1','w/ sensor, t=2','w/ sensor, t=5'})
% set(ch1,'FaceVertexCData',[1 0 1;0 0 0;])
% legend('Proposed Algorithm','COL Prior Algorithm','Path Greedy Algorithm');

%{
figure
cover_load = [1058 762 936 936 936;
              880 599 786 936 936;
              763 481 676 936 936;
             ];
pic_CoverLoad = bar(cover_load);
grid on;
ch4 = get(pic_CoverLoad,'children');
set(gca,'XTickLabel',{'No Loads','With Cell Phones','With AQI Sensors'})
% set(ch1,'FaceVertexCData',[1 0 1;0 0 0;])
legend('Proposed Algorithm','COL Prior Algorithm','Path Greedy Algorithm');
%}
