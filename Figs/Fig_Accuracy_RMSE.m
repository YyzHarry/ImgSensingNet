% ImgSensNet
% »­Í¼£ºFig1 + Fig2
%
% ±È½Ï×¼È·ÂÊ & RMSE
% Fig1£ºÖù×´Í¼ -- Acc
% Fig2£ºÖù×´Í¼ -- RMSE

%--------------------------------------- Fig 1 ------------------------------------%
%
ACUR_ALL = [97.11               % 3D CNN + feature
            91.66               % 2D CNN + feature
            77.56               % 2D CNN
            52.00               % DNN
            30.67               % SVM
            67.78               % kNN
            72.67               % DTree
            48.22               % MLR
            28.89               % RFC
           ];
b = bar(diag(ACUR_ALL),0.5,'stack');

% color = [[1 0 0.2];[0.9 0.6 0];[1 1 0];[0 1 0.4];[0 0.8 1];[0.9 0.4 0.8];[0.6 0 1]];
%{
b.CData(1,:) = [1 0 0.2];
b.CData(2,:) = [0.9 0.6 0];
b.CData(3,:) = [1 1 0];
b.CData(4,:) = [0 1 0.4];
b.CData(5,:) = [0 0.8 1];
b.CData(6,:) = [0.9 0.4 0.8];
b.CData(7,:) = [0.6 0 1];


for i = 1:9
    set(b(i),'FaceColor',color(i,:));
end
%}
grid on;
% set(gca,'XTickLabel',{'Proposed','DNN','kNN','CART','SVR','LI','MLR'})
% set(ch1,'FaceVertexCData',[1 0 1;0 0 0;])
% legend('Proposed Algorithm','Sequential Selection','Constraint Greedy Algorithm');



%--------------------------------------- Fig 2 ------------------------------------%
%%
figure
%
RMSE_ALL = [0.088               % 3D CNN + feature
            0.399               % 2D CNN + feature
            1.243               % 2D CNN
            6.442               % DNN
            7.432               % SVM
            3.770               % kNN
            3.589               % DTree
            5.542               % MLR
            7.392               % RFC
           ];
b = bar(diag(RMSE_ALL), 0.5, 'stack');

color = [[1 0 0];[0.9 0.6 0];[0.9 0.6 0];[0.9 0.6 0];[1 1 0];[0 1 0.4];[0 0.8 1];[0.9 0.4 0.8];[0.6 0 1]];
%{
b.CData(1,:) = [1 0 0.2];
b.CData(2,:) = [0.9 0.6 0];
b.CData(3,:) = [1 1 0];
b.CData(4,:) = [0 1 0.4];
b.CData(5,:) = [0 0.8 1];
b.CData(6,:) = [0.9 0.4 0.8];
b.CData(7,:) = [0.6 0 1];
%}

for i = 1:9
    set(b(i),'FaceColor',color(i,:));
end

grid on;
% set(gca,'XTickLabel',{'Proposed','DNN','kNN','CART','SVR','LI','MLR'})
% set(ch1,'FaceVertexCData',[1 0 1;0 0 0;])
% legend('Proposed Algorithm','Sequential Selection','Constraint Greedy Algorithm');
%
