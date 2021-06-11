%
%=========VisiLibity Demonstration Script=========
%
%This script uses the the MEX-files generated from
%visibility_polygon.cpp and in_environment.cpp.  Follow
%the instructions in the respective .cpp files to create
%these MEX-files before running this script. A graphical
%representation of the supplied environment file
%example1.environment is dislplayed and the user can
%then select points (must be in the environment) with
%the mouse and the visibility polygon of that point will
%be computed and plotted over the environment.
%


%Clear the desk
% clear all; close all; clc;
% format long;
% 
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




%Select test points and plot resulting visibility polygon

% current_x =[2,3,4,3,3,2,1,1,2,2,3,4,5,5,6,7,7,7]; % 
% current_y =[14,14,14,14,13,13,13,12,12,13,13,13,13,14,14,14,15,16];

% current_x = [4,4,4,5,6,7,6,7,6,7,6,7,6]; % 
% current_y = [6,7,8,8,8,8,8,8,8,8,8,8,8];
% 
% sensor_x = [4,4,4,4,5,6,7,6,7,6,7,6,7];
% sensor_y = [9,8,7,8,8,8,8,8,8,8,8,8,8];

current_x = Record_path_Agent(1,:);     %[5,4,3,2,1,1,1,2,3,3,3,4,5,6,7,8,8];
current_y = Record_path_Agent(2,:);     %[7,7,7,7,7,6,5,5,5,4,3,3,3,3,3,3,4];

sensor_x =  Record_path_Opponent(1,:);  %[5, 5,5,5,4,3,2,1,1,1,1,1,1,1,1,1,1];
sensor_y =  Record_path_Opponent(2,:);  %[10,9,8,7,7,7,7,7,6,5,4,5,6,7,6,5,4];

% current_x = [13,13,13,13,13,14,15,15];
% current_y = [ 5, 6, 7, 8, 9, 9, 9,10];
% 
% sensor_x =  [12,13,13,13,13,13,14,15];
% sensor_y =  [ 8, 8, 7, 7, 8, 9, 9, 9];

 Teammate1 =  Teammate;
%  Teammate2 = [10,11];

% Teammate3 = [13,6];
Total_scan = false(50,50);
reward_step = 0;
discount_factor = 0.8;

Ne_Total = 0;

for ii= 1: nnz(current_x)
    
    %Acquire test point.
    %[observer_x observer_y] = ginput(1);
    observer_x = current_x(ii);
    observer_y = current_y(ii);
    %Make sure the current point is in the environment
    if  in_environment( [observer_x observer_y] , environment , epsilon )
        
        %             Clear plot and form window with desired properties
        clf;  hold on;
        axis equal; axis off; axis([X_MIN X_MAX+4 Y_MIN Y_MAX+2]);
        
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
        
        if mod(ii,Teammate_appear_mod) == 0
            plot3(Teammate1(1),Teammate1(2), 0.3 , ...
                'p' , 'Markersize' , 16, 'MarkerFaceColor' , [0.9,0.8,0.7],'MarkerFaceColor','r','MarkerEdgeColor','r' );
%             plot3(Teammate2(1),Teammate2(2), 0.3 , ...
%                 'p' , 'Markersize' , 16, 'MarkerFaceColor' , [0.9,0.8,0.7],'MarkerFaceColor','r','MarkerEdgeColor','r' );
%                     plot3(Teammate3(1),Teammate3(2), 0.3 , ...
            %             'p' , 'Markersize' , 16, 'MarkerFaceColor' , [0.9,0.8,0.7],'MarkerFaceColor','r','MarkerEdgeColor','r' );
            %
        end
        %%overlap area
        x1= V{1}(:,1);
        y1= V{1}(:,2);
        b1 = poly2mask(x1,y1,50, 50);
        areaImage = bwarea(b1);
        Total_scan = b1 | Total_scan;
        reward_step(ii) = bwarea(Total_scan);
    end
    
    if  in_environment( [sensor_x(ii) sensor_y(ii)] , V , epsilon )
%         fprintf('\nThe robot is detected!\n');
%         t=text(6,13,1,'Robot Detected');
%         t(1).Color = 'r';
%         t(1).FontSize = 20;       
        sensor_detect_indicator(ii) = 1;
    else
        sensor_detect_indicator(ii)= 0;
    end
    
    
    if  in_environment( [Teammate1(1),Teammate1(2)] , W , epsilon ) && mod(ii,3) == 1 
        Teammate_detect_indicator1(ii) = 1;
    else
        Teammate_detect_indicator1(ii)= 0;
    end
    
    
%     if  in_environment( [Teammate2(1),Teammate2(2)] , W , epsilon )
%         Teammate_detect_indicator2(ii) = 1;
%     else
%         Teammate_detect_indicator2(ii)= 0;
%     end
%     
%     
%     if  in_environment( [Teammate3(1),Teammate3(2)] , W , epsilon )
%         Teammate_detect_indicator3(ii) = 1;
%     else
%         Teammate_detect_indicator3(ii)= 0;
%     end
   
     plot3(sensor_x(1:ii),sensor_y(1:ii),0.1*ones( nnz(sensor_x(1:ii)) , 1 ),'b','LineWidth',5)
     plot3(current_x(1:ii),current_y(1:ii),0.1*ones( nnz(current_x(1:ii)) , 1 ),'r','LineWidth',3)
     pause(0.1)
    txt = ['T = ',num2str(ii)];
    text(X_MAX/2-1,Y_MAX+1,txt,'FontSize',15)
    
    
    if sum(Teammate_detect_indicator1(1:ii)) > 0
        Teammate_detected1 = 1;
    else
        Teammate_detected1 = 0;
    end
    
        
%     if sum(Teammate_detect_indicator2(1:ii)) > 0
%         Teammate_detected2 = 1;
%     else
%         Teammate_detected2 = 0;
%     end
%     
%         
%     if sum(Teammate_detect_indicator3(1:ii)) > 0
%         Teammate_detected3 = 1;
%     else
%         Teammate_detected3 = 0;
%     end
    
    Ne_Total = Ne_Total + Teammate_detect_indicator1(ii)*Negtive_Teammate*discount_factor^sum(Teammate_detect_indicator1(1:ii));
%     Total_reward_step(ii) = reward_step(ii)- Negtive_Reward*sum(sensor_detect_indicator(1:ii))- Teammate_detected1*Negtive_Teammate - Teammate_detected2*Negtive_Teammate - Teammate_detected3*Negtive_Teammate;
%     if ii == 1
%         txt2 = ['Accumulated reward=',num2str(Total_reward_step(ii)), ', New Region Reward=',num2str(reward_step(ii))];
%     else
%         txt2 = ['Accumulated reward=',num2str(Total_reward_step(ii)), ', New Region Reward=',num2str(reward_step(ii) - reward_step(ii-1))];
%     end

%     text(X_MAX/2-5,Y_MAX,txt2,'FontSize',8)

    
    
    txt3 = [ 'Penalty(Detected)=',num2str(Negtive_Reward*sensor_detect_indicator(ii)),', Penalty(Teammate)=', num2str(Teammate_detect_indicator1(ii)*Negtive_Teammate*discount_factor^sum(Teammate_detect_indicator1(1:ii)-1))];
%        if sum(Teammate_detect_indicator(1:ii-1)) > 0
%              txt3 = [ 'Penalty(Detected)=',num2str(Negtive_Reward*sensor_detect_indicator(ii)),', Penalty(Teammate)=0'];
       if Teammate_detect_indicator1(ii)== 1
            txt3 = [ 'Penalty(Detected)=',num2str(Negtive_Reward*sensor_detect_indicator(ii)),', Penalty(Teammate)=10'];
       else
            txt3 = [ 'Penalty(Detected)=',num2str(Negtive_Reward*sensor_detect_indicator(ii)),', Penalty(Teammate)=0'];
       end
%     text(X_MAX/2-5,Y_MAX - 1,txt3,'FontSize',8)
    hold off
    
     mov(ii) = getframe(gca);
     jj = ii;
%      imwrite(mov(ii),sprintf('High%d.jpg',jj))
    %sensor the next point
    fname = sprintf('myfile%d.png', ii);
    saveas(gcf,fname)
%     
    
end

% v = VideoWriter('newfile.mp4','MPEG-4');
% v.FrameRate = 0.5;
% open(v)
% writeVideo(v,mov)
% close(v)
% hold on
%  plot3(sensor_x,sensor_y,0.1*ones( nnz(sensor_x) , 1 ),'b','LineWidth',5)
%  plot3(current_x,current_y,0.1*ones( nnz(current_x) , 1 ),'r','LineWidth',5)


 
for k = 1:ii
    if sum(Teammate_detect_indicator1(1:k)) > 0
        Teammate_detected1 = 1;
    else
        Teammate_detected1 = 0;
    end
    
    reward_step(k) = reward_step(k)- Negtive_Reward*sum(sensor_detect_indicator(1:k))- sum(Teammate_detect_indicator1)*Negtive_Teammate;
end

