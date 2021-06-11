function Vis = BuildMinimaxTree_BF2(Initial_Agent,Initial_Opponent,Initial_Agent_Region,Asset,Assets_Collected,environment,...
                                    Lookahead,Negtive_Reward,Negtive_Asset,Visibility_Data,Region,Assets_Detected,Asset_Visibility_Data,...
                                    Visibility_in_environment,step,Resolution,Discount_factor,epsilon)

environment_min_x = min(environment{1}(:,1));
environment_max_x = max(environment{1}(:,1));
environment_min_y = min(environment{1}(:,2));
environment_max_y = max(environment{1}(:,2));
X_MIN = floor(environment_min_x-0.1*(environment_max_x-environment_min_x));
X_MAX = floor(environment_max_x+0.1*(environment_max_x-environment_min_x));
Y_MIN = floor(environment_min_y-0.1*(environment_max_y-environment_min_y));
Y_MAX = floor(environment_max_y+0.1*(environment_max_y-environment_min_y));     
                                
Number_of_Asset = size(Asset,1);

Vis.Nodes.Agent{1} = Initial_Agent;
Vis.Nodes.Opponent{1} = Initial_Opponent;
Vis.Nodes.Generation = 1;

Vis.Nodes.Successors{1} = [];

% Each point will only be penalized for once
% assumes that X_min and Y_min are positive
assert(X_MIN >= 0, "X_min must be positive")
assert(Y_MIN >= 0, "Y_min must be positive")
Vis.Nodes.penalized{1}(X_MAX,Y_MAX) = 0;

%% Create the root node of the tree
Vis.Nodes.Assets_Detected{1} = Assets_Detected;
Vis.Nodes.Assets_Collected{1} = Assets_Collected;
Vis.Nodes.Agent_Region{1} = Initial_Agent_Region;

W{1} = Visibility_Data{Initial_Opponent(1) + X_MAX* Initial_Opponent(2)};
if in_environment( [Initial_Agent(1) Initial_Agent(2)] , W , epsilon )
    Vis.Nodes.Num_Times_Agent_Was_Detected = 1;
    Vis.Nodes.penalized{1}(Initial_Agent(1), Initial_Agent(2)) = 1;
else
    Vis.Nodes.Num_Times_Agent_Was_Detected = 0;
end
Vis.Nodes.Current_Step_reward_wo_assets = nnz(Vis.Nodes.Agent_Region{1})/Resolution^2 - Negtive_Reward* Vis.Nodes.Num_Times_Agent_Was_Detected(1);

for N = 1 : Number_of_Asset
    if in_environment( [Asset(N,1); Asset(N,2)] , W , epsilon )
        Vis.Nodes.Assets_Detected{1}(N) = 1;
    end
end


New_Initial = 1;
New_End = 1;
Count = 1;

Action_Space = [1 0; 0 1; -1 0; 0 -1; 0 0];

%% Start to build the search tree using breadth first expansion

for i = 2 : 2*Lookahead+1
    
    Initial_node = New_Initial;
    End_node = New_End;

    % Expand the MAX level, the agent's turn  
    if mod(i,2) == 0
        for j = Initial_node:End_node
            if j == Initial_node
                New_Initial = Count+1;
            end
            
            for action = 1:size(Action_Space,1)
                if Visibility_Data{ (Vis.Nodes.Agent{j}(1)+Action_Space(action,1)) + X_MAX*( Vis.Nodes.Agent{j}(2)+Action_Space(action,2))} ~= -1
                    % Add new edge to the tree
                    Vis.Nodes.Successors{j} = [ Vis.Nodes.Successors{j}, Count+1];
                    Vis.Nodes.Successors{Count+1} = [];
                    Vis.Nodes.Parent(Count+1) = j;
                    
                    % Update the agent's position
                    Vis.Nodes.Agent{Count+1} = [Vis.Nodes.Agent{j}(1)+Action_Space(action,1); Vis.Nodes.Agent{j}(2)+Action_Space(action,2)];
                    
                    % Opponent's position is the same as its parent node
                    Vis.Nodes.Opponent{Count+1} = Vis.Nodes.Opponent{j};
                    
                    % MAX level will not update detection times
                    Vis.Nodes.Num_Times_Agent_Was_Detected(Count+1) = Vis.Nodes.Num_Times_Agent_Was_Detected(j);
                    Vis.Nodes.penalized{Count+1} = Vis.Nodes.penalized{j};

                    % MAX level will not update assets detected/collected
                    Vis.Nodes.Assets_Detected(Count+1) = Vis.Nodes.Assets_Detected(j);
                    Vis.Nodes.Assets_Collected{Count+1} = Vis.Nodes.Assets_Collected{j}; 
                    
                    % MAX level will not update the positive reward
                    Vis.Nodes.Agent_Region{Count+1} =  Vis.Nodes.Agent_Region{j};
                    Vis.Nodes.Current_Step_reward_wo_assets(Count+1) =   Vis.Nodes.Current_Step_reward_wo_assets(j);

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
            
            for action = 1 : size(Action_Space,1)
                % Check the new point is in environment or not 
                 if Visibility_Data{(Vis.Nodes.Opponent{j}(1)+Action_Space(action,1)) + X_MAX*(Vis.Nodes.Opponent{j}(2)+Action_Space(action,2))} ~= -1
                    
                    % Add new edge to the tree
                    Vis.Nodes.Successors{j} = [Vis.Nodes.Successors{j}, Count+1];
                    Vis.Nodes.Successors{Count+1} = [];
%                     Vis.Nodes.Parent(Count+1) = j;                    
                    
                    % Agent's position is the same as its parent node
                    Vis.Nodes.Agent{Count+1} = Vis.Nodes.Agent{j};
                    
                    % Update the opponent's position
                    Vis.Nodes.Opponent{Count+1} = [Vis.Nodes.Opponent{j}(1)+Action_Space(action,1); Vis.Nodes.Opponent{j}(2)+Action_Space(action,2)];
                    
                    % Update the new area seen by the agent
                    Vis.Nodes.Agent_Region{Count+1} = Region{Vis.Nodes.Agent{Count+1}(1) + X_MAX*Vis.Nodes.Agent{Count+1}(2)} | Vis.Nodes.Agent_Region{j};
                    
                    % Update the agent detection penalty
                    Vis.Nodes.penalized{Count+1} = Vis.Nodes.penalized{j};
                    
                    
                    % Min level will update detection times, both for the agent and the assets  
                    % Check if the agent is in the opponent's visibility
                    % polygon AND has not previously been detected from the
                    % same position
                    W{1} = Visibility_Data{Vis.Nodes.Opponent{Count+1}(1) + X_MAX* Vis.Nodes.Opponent{Count+1}(2)};
                    if  Visibility_in_environment(Vis.Nodes.Agent{Count+1}(1) + X_MAX* Vis.Nodes.Agent{Count+1}(2), Vis.Nodes.Opponent{Count+1}(1) + X_MAX* Vis.Nodes.Opponent{Count+1}(2))...
                            &&  Vis.Nodes.penalized{Count+1}(Vis.Nodes.Agent{Count+1}(1),Vis.Nodes.Agent{Count+1}(2)) == 0
                        Vis.Nodes.Num_Times_Agent_Was_Detected(Count+1) = Vis.Nodes.Num_Times_Agent_Was_Detected(j) + 1;
                        Vis.Nodes.penalized{Count+1}(Vis.Nodes.Agent{Count+1}(1),Vis.Nodes.Agent{Count+1}(2)) = 1;
                    else
                        Vis.Nodes.Num_Times_Agent_Was_Detected(Count+1) = Vis.Nodes.Num_Times_Agent_Was_Detected(j);
                    end
                    
                    % Update the assets that are detected
                    Vis.Nodes.Assets_Detected(Count+1) = Vis.Nodes.Assets_Detected(j);
                    for N = 1 : Number_of_Asset
                        if Vis.Nodes.Assets_Detected{Count+1}(N) == 0
                            if Asset_Visibility_Data(N, Vis.Nodes.Opponent{Count+1}(1) + X_MAX* Vis.Nodes.Opponent{Count+1}(2)) == 1
                                Vis.Nodes.Assets_Detected{Count+1}(N) = 1;
                            end
                        end
                    end
                    
                    % Update the assets that are collected
                    Vis.Nodes.Assets_Collected{Count+1} = Vis.Nodes.Assets_Collected{j};
                    for N = 1 : Number_of_Asset
                        if  Asset(N,1) == Vis.Nodes.Opponent{Count+1}(1) &&  Asset(N,2) == Vis.Nodes.Opponent{Count+1}(2) && Vis.Nodes.Assets_Collected{Count+1}(N) == 0
                            Vis.Nodes.Assets_Collected{Count+1}(N) = step + (i-1)/2;
                        end
                    end
                                        
                    %Add discount factor
                    r =  (nnz(Vis.Nodes.Agent_Region{Count+1}) - nnz(Vis.Nodes.Agent_Region{j})) / (Resolution)^2 - Negtive_Reward * (Vis.Nodes.Num_Times_Agent_Was_Detected(Count+1) - Vis.Nodes.Num_Times_Agent_Was_Detected(j)) ;
                    
                    Vis.Nodes.Current_Step_reward_wo_assets(Count+1) =  Vis.Nodes.Current_Step_reward_wo_assets(j) + Discount_factor^((i-1)/2)*r;
                                                        
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