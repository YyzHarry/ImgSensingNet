% Fig1：地面——延迟间隔vs电量耐久度+精度
% Fig2：空中——飞行计划频率vs电量消耗+精度
% Fig3：空中——悬停时间vs电量消耗+精度（非单调）

close all;
figure(1);
set(gcf,'unit','normalized','position',[0,0,0.8,0.4]);

%--------------------------------------- Subplot 1 ------------------------------------%

subplot(1,3,1);

latency = 10:10:120;
duration = 13600./(50+(120./latency.*80));
deviation = linspace(98,90,12) - power((100-linspace(98,85,12)),2)/40 + 0.5*rand(1,12);
[ax,h1,h2]=plotyy(latency,duration,latency,deviation);
xlabel('Total Latency of Sensing & Uploading Intervals','FontSize',14);
ylabel(ax(1),'Battery Duration  (day)','FontSize',14);
ylabel(ax(2),'Accuracy  (%)','FontSize',14);
set(ax(1),'FontSize',14);
set(ax(2),'FontSize',14);
set(h1,'Marker','s','LineWidth',2,'MarkerSize',8);
set(h2,'Marker','v','LineWidth',2,'MarkerSize',8);
title('(a)');


%--------------------------------------- Subplot 2 ------------------------------------%

subplot(1,3,2);

flyinterval = 0.5:0.5:12;
powercomsume = 24./flyinterval;
deviation = 2+ linspace(5,20,24)*0.5 + power((linspace(5,20,24)),2)/50 + 0.5*rand(1,24);
[ax,h1,h2]=plotyy(flyinterval,powercomsume,flyinterval,deviation);
xlabel('Interval of Executing Aerial Sensing','FontSize',14);
ylabel(ax(1),'Power Consumption','FontSize',14);
ylabel(ax(2),'Relative Deviation  (%)','FontSize',14);
set(ax(1),'FontSize',14);
set(ax(2),'FontSize',14);
set(h1,'Marker','s','LineWidth',2,'MarkerSize',8);
set(h2,'Marker','v','LineWidth',2,'MarkerSize',8);
title('(b)');



%--------------------------------------- Subplot 3 ------------------------------------%

subplot(1,3,3);

hovertime = 0:1:20;
deviation = [7.092    6.5230    6.0829    5.7584    5.4550    5.3792    5.3612    5.6414    5.7310    5.9436    6.2903    6.5129 ...
    6.7906    6.9021    7.4074    7.9799    8.1904    8.7256    9.2752    9.6842    9.7259];
% deviation = 3 + (20-5*hovertime)./30 + 5./hovertime + hovertime/4+power(hovertime,2)/100+hovertime/30.*rand(1,21);
plot(hovertime,deviation,'-o','LineWidth',2,'MarkerSize',8,'Color','k');
xlabel('Hovering Time at Each Measure Position','FontSize',14);
ylabel('Relative Deviation  (%)','FontSize',14);
set(gca,'FontSize',14);
title('(c)');
