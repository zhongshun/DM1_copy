function Vis = BuildMinimaxTree_BF(Initial_Agent,Initial_Opponent,Initial_Agent_Region,Asset,Detection_Asset_Collect,environment,...
                                    Lookahead,Negtive_Reward,Negtive_Asset,Visibility_Data,Region,WiseUp_Index,Asset_Visibility_Data,...
                                    Visibility_in_environment,step,Resolution ,Discount_factor,epsilon)
                                
environment_min_x = min(environment{1}(:,1));
environment_max_x = max(environment{1}(:,1));
environment_min_y = min(environment{1}(:,2));
environment_max_y = max(environment{1}(:,2));
X_MIN = floor(environment_min_x-0.1*(environment_max_x-environment_min_x));
X_MAX = floor(environment_max_x+0.1*(environment_max_x-environment_min_x));
Y_MIN = floor(environment_min_y-0.1*(environment_max_y-environment_min_y));
Y_MAX = floor(environment_max_y+0.1*(environment_max_y-environment_min_y));     
                                

Number_of_Asset = size(Asset,1);
% Discount_factor = 1;


Vis.Nodes.Agent{1}= Initial_Agent;
Vis.Nodes.Opponent{1}=Initial_Opponent;
Vis.Nodes.Generation = 1;
Vis.Nodes.Successors{1} = [];

Vis.Nodes.panelized{1}(X_MAX,Y_MAX) = 0;
Vis.Nodes.Detection_Asset_WiseUp_Index{1} = WiseUp_Index;

Vis.Nodes.Detection_Asset_Collect{1} = Detection_Asset_Collect;

% Vis.Nodes.WiseUp = 0;

% V{1} = visibility_polygon( [Initial_Agent(1) Initial_Agent(2)] , environment , epsilon, snap_distance);
V{1} = Visibility_Data{Initial_Agent(1)+X_MAX*Initial_Agent(2)};
Vis.Nodes.Agent_Region{1} = Initial_Agent_Region;

W{1} = Visibility_Data{Initial_Opponent(1) +X_MAX* Initial_Opponent(2)};
if in_environment( [Initial_Agent(1) Initial_Agent(2)] , W , epsilon )
    Vis.Nodes.Agent_Detection_time = 1;
else
    Vis.Nodes.Agent_Detection_time = 0;
end
Vis.Nodes.Current_Step_reward_wo_assets = nnz(Vis.Nodes.Agent_Region{1}) / Resolution^2 - Negtive_Reward* Vis.Nodes.Agent_Detection_time(1);
% Vis.Nodes.Current_Step_reward_wo_assets_with_assest = Vis.Nodes.Current_Step_reward_wo_assets;
% Vis.Nodes.Asset_Collect_times(1) = 0;


for N = 1:Number_of_Asset
    if in_environment( [Asset(N,1) Asset(N,2)] , W , epsilon )
%         Vis.Nodes.Detection_Asset_WiseUp_Index{1}(N) = '1';
        Vis.Nodes.Detection_Asset_WiseUp_Index{1}(N) = 1;
    end
end


New_Initial = 1;
New_End = 1;
Count = 1;


Action_Space = [1 0;0 1;-1 0; 0 -1;0 0];

%% Start to build the search tree using breadth first expand
for i = 2:2*Lookahead+1
%     Current_step = ceil(i/2);
%     Negtive_Reward = (0.99^(i-1))*Negtive_Reward;
%     Negtive_Asset = (0.99^(i-1))*Negtive_Asset;
    Initial_node = New_Initial;
    End_node = New_End;

    % Expand the MAX level, the agent's turn  
    if mod(i,2) == 1
        for j = Initial_node:End_node
            if j == Initial_node
                New_Initial = Count+1;
            end
            
            for actions = 1:size(Action_Space,1)
                % Check the new point is in environment or not 
%                 if in_environment( [Vis.Nodes.Agent{j}(1)+Action_Space(actions,1), Vis.Nodes.Agent{j}(2)+Action_Space(actions,2)] , environment , 0.01 ) &&...
%                         in_environment( [Vis.Nodes.Agent{j}(1)+Action_Space(actions,1)*1/2, Vis.Nodes.Agent{j}(2)+Action_Space(actions,2)*1/2] , environment , 0.01 )
                if Visibility_Data{ (Vis.Nodes.Agent{j}(1)+Action_Space(actions,1)) + X_MAX*( Vis.Nodes.Agent{j}(2)+Action_Space(actions,2))} ~= -1
                    % Add new edge to the tree
%                     Vis=addedge(Vis,j,Count+1); ************
                    Vis.Nodes.Successors{j} = [ Vis.Nodes.Successors{j}, Count+1];
                    Vis.Nodes.Successors{Count+1} = [];
                    
                    % update the agent's position
                    Vis.Nodes.Agent{Count+1} = [Vis.Nodes.Agent{j}(1)+Action_Space(actions,1); Vis.Nodes.Agent{j}(2)+Action_Space(actions,2)];
                    % Opponent's position is the same as its parent node
                    Vis.Nodes.Opponent{Count+1} = Vis.Nodes.Opponent{j};
                    Vis.Nodes.panelized{Count+1} = Vis.Nodes.panelized{j};
                    % MAX level will  update detection times
                    W{1} = Visibility_Data{Vis.Nodes.Opponent{Count+1}(1) + X_MAX* Vis.Nodes.Opponent{Count+1}(2)};
                    %                     if in_environment( [Vis.Nodes.Agent{Count+1}(1) Vis.Nodes.Agent{Count+1}(2)] , W , epsilon )
                    if  Visibility_in_environment(Vis.Nodes.Agent{Count+1}(1) + X_MAX* Vis.Nodes.Agent{Count+1}(2), Vis.Nodes.Opponent{Count+1}(1) + X_MAX* Vis.Nodes.Opponent{Count+1}(2))...
                             && Vis.Nodes.panelized{Count+1}(Vis.Nodes.Agent{Count+1}(1),Vis.Nodes.Agent{Count+1}(2)) == 0
                        Vis.Nodes.Agent_Detection_time(Count+1) = Vis.Nodes.Agent_Detection_time(j) + 1;
                        Vis.Nodes.panelized{Count+1}(Vis.Nodes.Agent{Count+1}(1),Vis.Nodes.Agent{Count+1}(2)) = 1;
                    else
                        Vis.Nodes.Agent_Detection_time(Count+1) = Vis.Nodes.Agent_Detection_time(j);
                    end
                    
                    Vis.Nodes.Detection_Asset_Collect{Count+1} = Vis.Nodes.Detection_Asset_Collect{j};
                    Vis.Nodes.Detection_Asset_WiseUp_Index{Count+1} = Vis.Nodes.Detection_Asset_WiseUp_Index{j};
                    
                    for N = 1:Number_of_Asset
                        if  Asset(N,1) == Vis.Nodes.Opponent{Count+1}(1) &&  Asset(N,2) == Vis.Nodes.Opponent{Count+1}(2) && Vis.Nodes.Detection_Asset_Collect{Count+1}(N) == 0 ...
                            && Vis.Nodes.Detection_Asset_WiseUp_Index{Count+1}(N) == 1
%                             Vis.Nodes.Detection_Asset_Collect{Count+1}(N) = '1';
                            Vis.Nodes.Detection_Asset_Collect{Count+1}(N) = step + (i-1)/2;
%                             Vis.Nodes.Asset_Collect_times(Count+1) =  Vis.Nodes.Asset_Collect_times(j) + 1;
%                             Extra = X_MAX - step;
                        end
                    end
                    
                    
                    Vis.Nodes.Detection_Asset_WiseUp_Index(Count+1) = Vis.Nodes.Detection_Asset_WiseUp_Index(j);
%                     Vis.Nodes.Asset_Detection_time_E_smart(Count+1) = Vis.Nodes.Asset_Detection_time_E_smart(j);
                    Vis.Nodes.Parent(Count+1) = j;
                    % MAX level will  update the positive reward it
                    % collected
                    
                    V{1} = Visibility_Data{Vis.Nodes.Agent{Count+1}(1) + X_MAX*Vis.Nodes.Agent{Count+1}(2)};
                    Vis.Nodes.Agent_Region{Count+1} = Region{Vis.Nodes.Agent{Count+1}(1) + X_MAX*Vis.Nodes.Agent{Count+1}(2)} | Vis.Nodes.Agent_Region{j};
                    
%                     Vis.Nodes.Current_Step_reward_wo_assets(Count+1) =  nnz(Vis.Nodes.Agent_Region{Count+1})...
%                         - Negtive_Reward* Vis.Nodes.Agent_Detection_time(Count+1);
                    %Add discount factor
                    r =  (nnz(Vis.Nodes.Agent_Region{Count+1}) - nnz(Vis.Nodes.Agent_Region{j})) / Resolution^2 ...
                        - Negtive_Reward * (Vis.Nodes.Agent_Detection_time(Count+1) - Vis.Nodes.Agent_Detection_time(j)) ;
                    
                    Vis.Nodes.Current_Step_reward_wo_assets(Count+1) =  Vis.Nodes.Current_Step_reward_wo_assets(j) + Discount_factor^((i-1)/2)*r;

                    Vis.Nodes.Generation(Count+1) = i;
                    Count = Count+1;
                end
                
            end
            
            
            if j == End_node
                New_End = Count;
            end
            
        end
    % Expand the MIN level, the opponent's turn  
    else
        for j = Initial_node:End_node
            if j == Initial_node
                New_Initial = Count+1;
            end
            
            for actions = 1:size(Action_Space,1)
                % Check the new point is in environment or not 
                
                 if Visibility_Data{(Vis.Nodes.Opponent{j}(1)+Action_Space(actions,1)) + X_MAX*(Vis.Nodes.Opponent{j}(2)+Action_Space(actions,2))} ~= -1
                    % Add new edge to the tree
                    Vis.Nodes.Successors{j} = [Vis.Nodes.Successors{j}, Count+1];
                    Vis.Nodes.Successors{Count+1} = [];
                    
                    Vis.Nodes.panelized{Count+1} = Vis.Nodes.panelized{j};
                    % Agent's position is the same as its parent node
                    Vis.Nodes.Agent{Count+1} = Vis.Nodes.Agent{j};
                    Vis.Nodes.Opponent{Count+1} = [Vis.Nodes.Opponent{j}(1)+Action_Space(actions,1); Vis.Nodes.Opponent{j}(2)+Action_Space(actions,2)];
                    Vis.Nodes.Agent_Region{Count+1} = Vis.Nodes.Agent_Region{j};
                    Vis.Nodes.Agent_Detection_time(Count+1) = Vis.Nodes.Agent_Detection_time(j);
                    Vis.Nodes.Detection_Asset_WiseUp_Index(Count+1) = Vis.Nodes.Detection_Asset_WiseUp_Index(j);

                    Vis.Nodes.Detection_Asset_Collect{Count+1} = Vis.Nodes.Detection_Asset_Collect{j};
                    
                    Vis.Nodes.Current_Step_reward_wo_assets(Count+1) =  Vis.Nodes.Current_Step_reward_wo_assets(j);
%                     Vis.Nodes.Current_Step_reward_wo_assets_with_assest(Count+1) = Vis.Nodes.Current_Step_reward_wo_assets_with_assest(j);


                    
                    Vis.Nodes.Generation(Count+1) = i;
                    Count = Count+1;
                end
            end
            
            
            
            if j == End_node
                New_End = Count;
            end
            
        end
    end
    
    
end


end