function Reward_visbility = Heuristic2(Agent_Position, Opponent_Position, Agent_Region,Negtive_Reward,...
    Negtive_Asset,Assets,Assets_Collected,environment,Precompute_Path,PDF,WiseUp_Index,heur_agent_detection_weight,heur_agent_asset_weight,Visibility_Data)


environment_min_x = min(environment{1}(:,1));
environment_max_x = max(environment{1}(:,1));
environment_min_y = min(environment{1}(:,2));
environment_max_y = max(environment{1}(:,2));
X_MIN = floor(environment_min_x-0.1*(environment_max_x-environment_min_x));
X_MAX = floor(environment_max_x+0.1*(environment_max_x-environment_min_x));
Y_MIN = floor(environment_min_y-0.1*(environment_max_y-environment_min_y));
Y_MAX = floor(environment_max_y+0.1*(environment_max_y-environment_min_y)); 


%Compute the positive reward
Reward_visbility = 0;
for x = max(floor(X_MIN),1):floor(X_MAX)+1
    for y = max(floor(Y_MIN),1):floor(Y_MAX)+1
        
        % if a point is in the environment and not yet been seen by the
        % agent

        %if in_environment( [x,y] , environment , epsilon ) && ~Agent_Region{1}(x,y)
        if Visibility_Data{X_MAX*y + x} ~= -1 & ~Agent_Region{1}(x,y)
            D = Precompute_Path{X_MAX*Agent_Position(2)+Agent_Position(1), X_MAX*y + x};
            Reward_visbility = Reward_visbility + PDF(length(D(1,:)));
        end
    end
end

%Compute the negative reward of the opponent detecting the agent

D = Precompute_Path{X_MAX*Agent_Position(2) + Agent_Position(1), X_MAX*Opponent_Position(2) + Opponent_Position(1)};
Penalty_Opponent = PDF(length(D(1,:)));

%Compute the negative reward of the asset
Penalty_Asset = 0;
for L = 1:length(Assets(:,1))
    if Assets_Collected(L) == 1 || WiseUp_Index(L) == 0
        continue;
    end
    
    D = Precompute_Path{X_MAX*Opponent_Position(2)+Opponent_Position(1), X_MAX*Assets(L,2)+Assets(L,1)};
    Penalty_Asset = Penalty_Asset + PDF(length(D(1,:)));
end

Reward_visbility = Reward_visbility - heur_agent_detection_weight*Negtive_Reward*Penalty_Opponent - heur_agent_asset_weight*Negtive_Asset*Penalty_Asset;
end