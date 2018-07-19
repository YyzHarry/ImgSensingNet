% ImgSensNet
% 画图：Runtime comparison
%
% Fig1：Wakeup Schedule Design, 随r增加: 算法运行时间/复杂度

%-------------------------------------- Wakeup Schedule --------------------------------------%
%
coord = load('LocatCoordinates.txt');
coord_x = coord(1,:);
coord_y = coord(2,:);
% Point-of-Interest (5)
poiX = [160, 300, 60, 490, 510];
poiY = [10, 220, 370, 60, 360];

% Radius (hyperparam)
r = 100;

graph = zeros(30,30);
degree_index = zeros(30,30);   % 每行前degree个数为与index相连的点的index
degree = zeros(1,30);
nodes_num = 30;
min_idpt_domin_set_num = 0;

tic;    % 开始计时
for i=1:30
    for j=(i+1):30
        if (sqrt((coord_x(i)-coord_x(j))^2 + (coord_y(i)-coord_y(j))^2) <= r) && (i~=j)
            graph(i,j) = 1;
            graph(j,i) = 1;
            degree(i) = degree(i)+1;
            degree(j) = degree(j)+1;
            degree_index(i,degree(i)) = j;
            degree_index(j,degree(j)) = i;
        end
    end
end

while nodes_num ~= 0
    [max_val, max_index] = max(degree);
    min_idpt_domin_set_num = min_idpt_domin_set_num+1;
    for i=1:30    % 选出第i个不是0的，即相连的index
        if degree_index(max_index,i) == 0
            continue
        end
        for j=1:30
            if graph(degree_index(max_index,i),j)  && j~=max_index
                graph(degree_index(max_index,i),j) = 0;
                graph(j,degree_index(max_index,i)) = 0;
                degree_index(degree_index(max_index,i), find(degree_index(degree_index(max_index,i), :) == j)) = 0;
                degree_index(j, find(degree_index(j, :) == degree_index(max_index,i))) = 0;
                degree(j) = degree(j)-1;
            end
        end
        degree(degree_index(max_index,i)) = 0;
    end
    
    degree(max_index) = 0;
    nodes_num = nodes_num - max_val - 1;
end

disp(['Wake-up算法计算 r=',num2str(r),' 时的运行时间：',num2str(toc)]);
disp(['r=',num2str(r),' 时最终需要wake up的设备数目：',num2str(min_idpt_domin_set_num)])
%}


%-------------------------------------- Fig1 --------------------------------------%
%
device_on_30 = [28.8               % r = 50
                25.3               % 100
                23.9               % 200
                16.8               % 300
                ];
bar(diag(device_on_30),0.5,'stack');

figure
running_time_30 = [0.012341               % r = 50
                   0.011866               % 100
                   0.011254               % 200
                   0.007937               % 300
                   ];
bar(diag(running_time_30),0.5,'stack');

figure
device_on_100 = [96.8               % r = 50
                 81.4               % 100
                 60.9               % 200
                 36.8               % 300
                 ];
bar(diag(device_on_100),0.5,'stack');

figure
running_time_100 = [0.82341               % r = 50
                    0.51866               % 100
                    0.41254               % 200
                    0.27937               % 300
                    ];
bar(diag(running_time_100),0.5,'stack');
