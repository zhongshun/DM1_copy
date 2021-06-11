clear all; 
clc;

set_params; % set all parameters in this file

%% Create environment

environment = read_vertices_from_file('./Environments/Env_A.environment');

% Asset = [9 6;9 4];
Asset = [5 1];
Number_of_Asset = size(Asset,1);

Initial_Agent = [5;3];%[8;5];
Initial_Opponent = [7;3];%[7;5];

Creat_Environment_Visbility_Data

%% Initialization

Record_path_Agent = Initial_Agent;
Record_path_Opponent = Initial_Opponent;


Assets_Collected = zeros(Number_of_Asset,1);

V{1} = visibility_polygon( [Initial_Agent(1) Initial_Agent(2)] , environment , epsilon, snap_distance);
Initial_Agent_Region = poly2mask(Resolution*V{1}(:,1),Resolution*V{1}(:,2),Resolution*ENV_SIZE1, Resolution*ENV_SIZE2);

Number_of_Function = 0;
for i = 0:Number_of_Asset
    Number_of_Function = Number_of_Function + nchoosek(Number_of_Asset,i);
end
Function_index = dec2bin(Number_of_Function-1);
Function_index_size = size(Function_index,2);
Assets_Detected = zeros(Number_of_Asset,1);


%% Run the episode
for step = 1:T_execution
    
    %% Build the tree
    if T_execution - step + 1  <= Lookahead
        Lookahead = T_execution - step + 1;
    end     

    Tree_Agent = BuildMinimaxTree_BF2(Initial_Agent,Initial_Opponent,Initial_Agent_Region,Asset,...
            Assets_Collected,environment,Lookahead,Negtive_Reward,Negtive_Asset,Visibility_Data,Region,Assets_Detected,Asset_Visibility_Data,Visibility_in_environment,step,Resolution,Discount_factor,epsilon);
        
    %% Run the DM1 One Pass to back propagate the reward values
    % Change RunDM1 to RunLeafLookAhed or RunMinimax_multi_assets to run
    % other algorithms
    
    [Initial_Agent_update,Initial_Opponent1,Initial_Agent_Region_update,Assets_Collected_agent] = ...
        RunDM1(Tree_Agent,Lookahead,Asset,Negtive_Reward,Negtive_Asset,Number_of_Function,Function_index_size,Visibility_Data,Region,Asset_Visibility_Data,step,Discount_factor,environment,Precompute_Path,Assets_Detected,heur_penalty_std,heur_agent_detection_weight,epsilon);
    clear Tree_Agent;
%     
    %% Build the tree for the opponent 
    Tree_Opponent = BuildMinimaxTree_BF(Initial_Agent,Initial_Opponent,Initial_Agent_Region,Asset,...
            Assets_Collected,environment,Lookahead,Negtive_Reward,Negtive_Asset,Visibility_Data,Region,Assets_Detected,Asset_Visibility_Data,Visibility_in_environment,step,Resolution,Discount_factor);
    
    %% Run the Minimax
    [Initial_Agent1,Initial_Opponent_update,Initial_Agent_Region_opponent,Assets_Collected] = ...
        RunMinimax(Tree_Opponent,T,Asset,Negtive_Reward,Negtive_Asset,Number_of_Function,Function_index_size,Visibility_Data,Region,Asset_Visibility_Data,Visibility_in_environment,step,Discount_factor,environment,Precompute_Path,Assets_Detected);
    clear Tree_Opponent;

    %% Record the action for next step, also record the assets collected realdy
    Record_path_Agent(:,step + 1) = Initial_Agent_update;
    Record_path_Opponent(:,step + 1) = Initial_Opponent_update;
    Initial_Agent = Initial_Agent_update;
    Initial_Opponent = Initial_Opponent_update;
    Initial_Agent_Region = Initial_Agent_Region_update;
    Assets_Collected = Assets_Collected;
    
    W{1} = Visibility_Data{Initial_Opponent(1) +100* Initial_Opponent(2)};
    for N = 1:Number_of_Asset
        if in_environment( [Asset(N,1) Asset(N,2)] , W , epsilon )
            Assets_Detected(N) = 1;
        end
    end

end

%%
save('Online.mat')

%%
Plot_Path_DM1