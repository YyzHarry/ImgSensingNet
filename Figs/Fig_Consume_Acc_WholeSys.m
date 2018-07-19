% ImgSensNet
% 画图：Whole System Performance (i.e., Acc, Consume)
%
% Fig1：多折线; 不同JE影响 vs. Remained energy
% Fig2：多折线; 不同AQI影响 vs. Acc & Consume; Subplot
% Fig3：2折线; Trade-off between Acc & Consume
% Fig4：Consume与其他对比 * 2 (UAV + WSN)

%--------------------------------------- Fig 1 ------------------------------------%
%
figure
x = 0:5:30;
x_je02 = 0:5:25;
x_je0 = 0:5:20;

je_1 = [100   99.2123   98.0012   96.5423   94.5110   92.0019   89.1293];
je_08 = [100   96.9877   91.1149   84.3421   75.1723   65.0167   51.2485];
je_06 = [100   93.6101   85.6629   75.2730   63.0860   49.6268   32.2357];
je_04 = [100   90.7677   79.1753   65.9492   50.0247   33.0282   13.6218];
je_02 = [100   87.0327   73.1727   57.5626   39.2759   19.8167];
je_0 = [100   84.1496    65.7151   42.1043   15.7409];

plot(x, je_1 ,'-*',...
                'LineWidth',2,...
                'Color',[0 0 0],...
                'MarkerSize',6);
%
hold on
plot(x, je_08 ,'-^',...
                'LineWidth',2,...
                'Color',[0.9 0.3 0.1],...
                'MarkerSize',6);
%
hold on
plot(x, je_06, '-s',...
                'LineWidth',2,...
                'Color',[0.4 0.6 0.8],...
                'MarkerSize',6);
%
hold on
plot(x, je_04, '-s',...
                'LineWidth',2,...
                'Color',[0.4 0.7 0.2],...
                'MarkerSize',6);
%
hold on
plot(x_je02, je_02, '-o',...
                'LineWidth',2,...
                'Color',[0.1 0.3 0.9],...
                'MarkerSize',6);
%
hold on
plot(x_je0, je_0, '-o',...
                'LineWidth',2,...
                'Color',[0.1 0.3 0.9],...
                'MarkerSize',6);
%
axis([0 30 0 100]),
set(gca,'xgrid','on'), set(gca,'ygrid','on')
grid on
xlabel('Time (days)'), ylabel('Remained energy (%)');
%

%--------------------------------------- Fig 2 ------------------------------------%
%
figure
x = 0:0.1:0.5;

low_acc = [85.09   84.02   80.51   74.11   73.32   71.64];
mid_acc = [92   90.9877   89.1149   82.3421   82.1723   80.0167];
high_acc = [95   93.6101   93.3629   87.2730   85.0860   84.6268];
low_consume = [87   86.5   85.1753   79.9492   72.0247   62.0282];
mid_consume = [95   94.0327   92.1727   87.5626   82.2759   73.8167];
high_consume = [99   95.1496   94.7151   89.1043   82.7409   76];

subplot(121)
plot(x,low_acc,'-v',...
              'LineWidth',2,...
              'Color',[0.9 0.3 0.1],...
              'MarkerSize',6);
hold on
plot(x,mid_acc,'-s',...
              'LineWidth',2,...
              'Color',[0.4 0.7 0.2],...
              'MarkerSize',6);
hold on
plot(x,high_acc,'-o',...
               'LineWidth',2,...
               'Color',[0.5 0 0.6],...
               'MarkerSize',6);

axis([0 0.5 60 100]),
set(gca,'xgrid','on'), set(gca,'ygrid','on')
grid on
xlabel('JE'), ylabel('Estimation accuracy (%)');

subplot(122)
plot(x,low_consume,'--v',...
              'LineWidth',2,...
              'Color',[0.9 0.3 0.1],...
              'MarkerSize',6);
hold on
plot(x,mid_consume,'--s',...
              'LineWidth',2,...
              'Color',[0.4 0.7 0.2],...
              'MarkerSize',6);
hold on
plot(x,high_consume,'--o',...
               'LineWidth',2,...
               'Color',[0.5 0 0.6],...
               'MarkerSize',6);

axis([0 0.5 60 100]),
set(gca,'xgrid','on'), set(gca,'ygrid','on')
grid on
xlabel('JE'), ylabel('Normalized consumption (%)');
%

%--------------------------------------- Fig 3 ------------------------------------%
%
figure
set(gcf,'unit','normalized','position',[0,0,0.5,0.5]);
je_x = 0:0.1:1;
powercomsume = [95.5   94.1496   92.7151   85.1043   78.7409   70.1   59.58   45.12   27.29   17.23   12.12];
deviation = [5.5   6.1   7   8.5   11.7409   18.1   25.58   32.12   39.29   44.13   47.12];
[ax, h1, h2] = plotyy(je_x, powercomsume, je_x, deviation);
set(ax(1), 'FontSize', 16, 'FontName', 'Times');
set(ax(2), 'FontSize', 16, 'FontName', 'Times');
xlabel('JE', 'FontSize', 24, 'FontName', 'Times');
ylabel(ax(1), 'Normalized consumption (%)', 'FontSize', 24, 'FontName', 'Times');
ylabel(ax(2), 'RMSE \xi', 'FontSize', 24, 'FontName', 'Times');
set(h1, 'Marker', 's', 'LineWidth', 3, 'MarkerSize', 12);
set(h2, 'Marker', 'v', 'LineWidth', 3, 'MarkerSize', 12);
% title('(b)');
%

%--------------------------------------- Fig 4 ------------------------------------%
%--------------------------------------- UAV
%
figure
time_load = [11 21 32;
             23 42 58;
             49 78 96;
            ];
pic_TimeLoad = bar(time_load);
grid on;
ch2 = get(pic_TimeLoad,'children');
set(gca,'XTickLabel',{'20','50','100'})
legend('ImgSensingNet','ARMS, t=2s','ARMS, t=5s');
xlabel('Coverage space (*10^3 m^3)'), ylabel('Normalized consumption (%)');

%--------------------------------------- WSN
%
figure
cover_no_load = [18 35 37;
                 29 56 62;
                 54 89 97;
                ];
pic_CoverNoLoad = bar(cover_no_load);
grid on;
ch3 = get(pic_CoverNoLoad,'children');
set(gca,'XTickLabel',{'1','2','5'})
legend('ImgSensingNet (JE=0.5)','ImgSensingNet (JE=0)','AQNet');
xlabel('Coverage space (km^2)'), ylabel('Normalized consumption (%)');
%
