% load('Online.mat');
% load('Save_Visibility_Data\M_starstar12.mat')
load('Save_Visibility_Data\Show_Tree.mat')

% environment = read_vertices_from_file('./Environments/M_starstar12.environment');

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






current_x = Agent_path_x;    
current_y = Agent_path_y;     

sensor_x =  Opponent_path_x; 
sensor_y =  Opponent_path_y;  


Asset = Asset_Position;

Teammate = Asset_Position; 
TeammatePenalty = Negtive_Asset;
Teammate_detected = zeros(size(Asset_Position,1),1);





for ii= 1: 1
    

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
        
        
%         W{1} = visibility_polygon( [Opponent_path_x(1) Opponent_path_y(1)] , environment , 0.001*epsilon , snap_distance );
%         V{1} = visibility_polygon( [Agent_path_x(1) Agent_path_y(1)] , environment , 0.001*epsilon , snap_distance );
%         
%         
%         %sensor polygon
%         
%         Area_sensor = polyarea(W{1}(:,1),W{1}(:,2));
%         patch( W{1}(:,1) , W{1}(:,2) , 0.1*ones( size(W{1},1) , 1 ) , ...
%             [0.7,0.7,0.9] , 'LineStyle' , 'none' );
%         %         plot3( W{1}(:,1) , W{1}(:,2) , 0.1*ones( size(W{1},1) , 1 ) , ...
%         %             'y*' , 'Markersize' , 5 );
        plot3( sensor_x(ii) , sensor_y(ii) , 0.3 , ...
            's' , 'Markersize' , 15, 'MarkerFaceColor' , [0.9,0.8,0.7],'MarkerFaceColor','b','MarkerEdgeColor','b' );
        
        
        
        %Compute and plot visibility polygon
        
%         Area = polyarea(V{1}(:,1),V{1}(:,2));
%         
%         vpatch= patch( V{1}(:,1) , V{1}(:,2) , 0.1*ones( size(V{1},1) , 1 ) , ...
%             [0.9,0.5,0.5],'LineStyle' , 'none' );
%         %         plot3( V{1}(:,1) , V{1}(:,2) , 0.1*ones( size(V{1},1) , 1 ) , ...
%         %             'b*' , 'Markersize' , 5 );
%         alpha(vpatch, 0.6)
        
        
        hold on
              
        
    end
    
        
        txt1 = ['DM1 Current step = ',num2str(step)];
        text(X_MAX/2-1,Y_MAX+5,txt1,'FontSize',20)
        
        txt2 = ['Agent      X: ',num2str(Agent_path_x)];
        text(X_MAX/2-3,Y_MAX+4,txt2,'FontSize',20)
        
        txt3 = ['Agent      Y: ',num2str(Agent_path_y)];
        text(X_MAX/2-3,Y_MAX+3,txt3,'FontSize',20)
        
        txt4 = ['Oppoent  X: ',num2str(Opponent_path_x)];
        text(X_MAX/2-3,Y_MAX+2,txt4,'FontSize',20)
        
        txt5 = ['Oppoent  Y: ',num2str(Opponent_path_y)];
        text(X_MAX/2-3,Y_MAX+1,txt5,'FontSize',20)
        
        CurrentPenalty = 0;
    

    
end
hold on;
plot3(Opponent_path_x(1:length(Opponent_path_x)),Opponent_path_y(1:length(Opponent_path_y)),0.1*ones( length(Opponent_path_y) , 1 ),'b','LineWidth',5)
%     plot3(sensor_x(ii-1:ii),sensor_y(ii-1:ii),0.1*ones( max(size(sensor_x(ii-1:ii))) , 1 ),':b','LineWidth',5)
plot3(Agent_path_x(1:length(Agent_path_x)),Agent_path_y(1:length(Agent_path_y)),0.1*ones( length(Agent_path_x) , 1 ),'r','LineWidth',5);

for k = 1:size(Asset,1)
    plot3(Asset(k,1),Asset(k,2), 0.3 , ...
        'p' , 'Markersize' , 16, 'MarkerFaceColor' , [0.9,0.8,0.7],'MarkerFaceColor','r','MarkerEdgeColor','r' );
end

    hold off
    
         mov(ii) = getframe(gca);
         jj = ii;
    %      imwrite(mov(ii),sprintf('High%d.jpg',jj))
    
        %sensor the next point
        fname = sprintf('save_figure/DM1Tree%d.png', step);
        saveas(gcf,fname)
    %
   



