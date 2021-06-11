load('Online.mat');


environment_min_x = min(environment{1}(:,1));
environment_max_x = max(environment{1}(:,1));
environment_min_y = min(environment{1}(:,2));
environment_max_y = max(environment{1}(:,2));
X_MIN = environment_min_x-0.1*(environment_max_x-environment_min_x);
X_MAX = environment_max_x+0.1*(environment_max_x-environment_min_x);
Y_MIN = environment_min_y-0.1*(environment_max_y-environment_min_y);
Y_MAX = environment_max_y+0.1*(environment_max_y-environment_min_y);



%Clear plot and form window with desired properties
clf; hold on;
axis equal; axis off; axis([X_MIN X_MAX+4 Y_MIN Y_MAX]);


%Plot Environment
patch( environment{1}(:,1) , environment{1}(:,2) , 0.1*ones(1,length(environment{1}(:,1)) ) , ...
    'w' , 'linewidth' , 1.5 );
for i = 2 : size(environment,2)
    patch( environment{i}(:,1) , environment{i}(:,2) , 0.1*ones(1,length(environment{i}(:,1)) ) , ...
        'k' , 'EdgeColor' , [0 0 0] , 'FaceColor' , [0.8 0.8 0.8] , 'linewidth' , 1.5 );
end






current_x = Record_path_Agent(1,:);    
current_y = Record_path_Agent(2,:);     

sensor_x =  Record_path_Opponent(1,:); 
sensor_y =  Record_path_Opponent(2,:);  




Teammate = Asset; 
Teammate_detected = zeros(1,5);
TeammatePenalty = Negtive_Asset;
Teammate_detected = zeros(size(Asset,1),1);

Updated_Negtive_Reward = Negtive_Reward;



% Total_scan = false(Resolution*ENV_SIZE1, Resolution*ENV_SIZE2);
reward_step = 0;
Total_scan = false(Resolution*ENV_SIZE1, Resolution*ENV_SIZE2);
reward_step = 0;
CurrentPenalty = 0;


V{1} = visibility_polygon( [current_x(1) current_y(1)] , environment , epsilon , snap_distance );
W{1} = visibility_polygon( [sensor_x(1) sensor_y(1)] , environment , epsilon , snap_distance );
Total_visiable{1} =  V;

Ne_Total = 0;

if  in_environment( [sensor_x(1) sensor_y(1)] , V , epsilon )
    
    sensor_detect_indicator(1) = 1;
else
    sensor_detect_indicator(1)= 0;
end

for k = 1:size(Asset,1)
    plot3(Asset(k,1),Asset(k,2), 0.3 , ...
        'p' , 'Markersize' , 16, 'MarkerFaceColor' , [0.9,0.8,0.7],'MarkerFaceColor','r','MarkerEdgeColor','r' );
end



for ii= 1: max(size(current_x))
    
    TeammatePenalty = Negtive_Asset;
    Updated_Negtive_Reward = Negtive_Reward;
    
    observer_x = current_x(ii);
    observer_y = current_y(ii);
    %Make sure the current point is in the environment
    if  in_environment( [observer_x observer_y] , environment , epsilon )
        
        %             Clear plot and form window with desired properties
        clf;  hold on;
        axis equal;
        axis off; axis([X_MIN X_MAX Y_MIN Y_MAX+6]);
        
        %Plot environment
        patch( environment{1}(:,1) , environment{1}(:,2) , 0.1*ones(1,length(environment{1}(:,1)) ) , ...
            'w' , 'linewidth' , 1.5 );
        for i = 2 : size(environment,2)
            patch( environment{i}(:,1) , environment{i}(:,2) , 0.1*ones(1,length(environment{i}(:,1)) ) , ...
                'k' , 'EdgeColor' , [0 0 0] , 'FaceColor' , [0.8 0.8 0.8] , 'linewidth' , 0.1 );
        end
        
        
        
        %             Plot observer
        plot3( observer_x , observer_y , 0.3 , ...
            'o' , 'Markersize' , 15 , 'MarkerEdgeColor' , 'k' , 'MarkerFaceColor' , 'r' );
        hold on
        
        
        W{1} = visibility_polygon( [sensor_x(ii) sensor_y(ii)] , environment , epsilon , snap_distance );
        V{1} = visibility_polygon( [observer_x observer_y] , environment , epsilon , snap_distance );
        
        
        %sensor polygon
        
        Area_sensor = polyarea(W{1}(:,1),W{1}(:,2));
        patch( W{1}(:,1) , W{1}(:,2) , 0.1*ones( size(W{1},1) , 1 ) , ...
            [0.7,0.7,0.9] , 'LineStyle' , 'none' );
        %         plot3( W{1}(:,1) , W{1}(:,2) , 0.1*ones( size(W{1},1) , 1 ) , ...
        %             'y*' , 'Markersize' , 5 );
        plot3( sensor_x(ii) , sensor_y(ii) , 0.3 , ...
            's' , 'Markersize' , 15, 'MarkerFaceColor' , [0.9,0.8,0.7],'MarkerFaceColor','b','MarkerEdgeColor','b' );
        
        
        %total polygon
        
        Total_visiable{ii} =  V;
        
        for k = 1:ii-1
            tpatch = patch( Total_visiable{k}{1}(:,1) , Total_visiable{k}{1}(:,2) , 0.1*ones( size(Total_visiable{k}{1},1) , 1 ) , ...
                [0.9,0.8,0.8] , 'LineStyle' , 'none' );
            alpha(tpatch,0.6)
        end
        
        %Compute and plot visibility polygon
        
        Area = polyarea(V{1}(:,1),V{1}(:,2));
        
        vpatch= patch( V{1}(:,1) , V{1}(:,2) , 0.1*ones( size(V{1},1) , 1 ) , ...
            [0.9,0.5,0.5],'LineStyle' , 'none' );
        %         plot3( V{1}(:,1) , V{1}(:,2) , 0.1*ones( size(V{1},1) , 1 ) , ...
        %             'b*' , 'Markersize' , 5 );
        alpha(vpatch, 0.6)
        
        
        hold on
        
        for k = 1:size(Asset,1)
            if sensor_x(ii) == Asset(k,1) && sensor_y(ii) ==  Asset(k,2)
                Teammate_detected(k) = 1;
                CurrentPenalty = 1;
            end
        end

        for k = 1:size(Asset,1)
            if Teammate_detected(k) == 0
                plot3(Asset(k,1),Asset(k,2), 0.3 , ...
                    'p' , 'Markersize' , 16, 'MarkerFaceColor' , [0.9,0.8,0.7],'MarkerFaceColor','r','MarkerEdgeColor','r' );
            end
        end

        



        %%overlap area
        x1= V{1}(:,1);
        y1= V{1}(:,2);
        b1 = poly2mask(Resolution*x1,Resolution*y1,Resolution*ENV_SIZE1, Resolution*ENV_SIZE2);
        areaImage = bwarea(b1);
        Total_scan = b1 | Total_scan;
        reward_step(ii) = bwarea(Total_scan)/Resolution^2;
        

        
        
              
        
    end
    
    
    if  in_environment( [sensor_x(ii) sensor_y(ii)] , V , epsilon )
        sensor_detect_indicator(ii) = Updated_Negtive_Reward;
    else
        sensor_detect_indicator(ii)= 0;
    end
    if ii >= 2
        
        txt1 = ['T = ',num2str(ii)];
        text(X_MAX/2-1,Y_MAX+5,txt1,'FontSize',20)
        
        txt2 = ['Region Exploration:  Total Reward=',num2str(reward_step(ii)), ', Current Reward =',num2str(reward_step(ii) - reward_step(ii-1))];
        text(X_MIN+4,Y_MAX+4,txt2,'FontSize',20)
        
        txt3 = ['Agent Observation: Total Penalty = ',num2str(sum(sensor_detect_indicator(1:ii))),', Current Penalty =', num2str(sensor_detect_indicator(ii))];
        text(X_MIN+4,Y_MAX+3,txt3,'FontSize',20)
        
        txt4 = ['Assets Captured: Total Penalty = ',num2str(TeammatePenalty*sum(Teammate_detected)),', Current Penalty =', num2str(TeammatePenalty*CurrentPenalty)];
        text(X_MIN+4,Y_MAX+2,txt4,'FontSize',20)
        
        txt5 = ['Combined: Total Reward = ',num2str(reward_step(ii)- sum(sensor_detect_indicator(1:ii))...
            -(TeammatePenalty*sum(Teammate_detected))),', Current Reward =', num2str(reward_step(ii) - reward_step(ii-1)- ...
            (sensor_detect_indicator(ii)) - (TeammatePenalty*CurrentPenalty))];
        text(X_MIN+4,Y_MAX+1,txt5,'FontSize',20)
        
        CurrentPenalty = 0;
    
    
    plot3(sensor_x(1:ii-1),sensor_y(1:ii-1),0.1*ones( max(size(sensor_x(1:ii-1))) , 1 ),'b','LineWidth',5)
    plot3(sensor_x(ii-1:ii),sensor_y(ii-1:ii),0.1*ones( max(size(sensor_x(ii-1:ii))) , 1 ),':b','LineWidth',5)
    plot3(current_x(1:ii-1),current_y(1:ii-1),0.1*ones( max(size(current_x(1:ii-1))) , 1 ),'r','LineWidth',5)
    plot3(current_x(ii-1:ii),current_y(ii-1:ii),0.1*ones( max(size(current_x(ii-1:ii))) , 1 ),':r','LineWidth',5)
    pause(0.1)
    end
    hold off
    %
         mov(ii) = getframe(gca);
         jj = ii;
    %      imwrite(mov(ii),sprintf('High%d.jpg',jj))
    
        %sensor the next point
        fname = sprintf('save_figure/DM1%d.png', ii);
        saveas(gcf,fname)
    % %
%    
    
end




