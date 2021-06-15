epsilon = 0.000000001;  %Robustness constant
snap_distance = 0.05;   %Snap distance (distance within which an observer location will be snapped to the
                        %boundary before the visibility polygon is computed)
                        
Number_of_Assets = 1;     

Number_of_Experiments = 1;

Resolution = 1;         % 1 is a coarse grid used for computing visibility

Negtive_Reward = 0.3;   % penalty for agent being detected
Negtive_Asset = 30;     % penalty for asset being detected

Lookahead = 5;          % planning horizon        

T_execution = 10;       % episode duration
Discount_factor = 0.93;

USE_HEURISTIC = false;         
heur_penalty_std = 1;        % std of penalty to use for the heuristic
heur_agent_asset_weight = 1;            % how much weight to give the asset penalty in the heuristic
heur_agent_detection_weight = 1;     % how much weight to give the opponent detecting the agent penalty in the heuristic

environment = read_vertices_from_file('./Environments/Env_A.environment');  %The environment and its boundary
environment_min_x = min(environment{1}(:,1));
environment_max_x = max(environment{1}(:,1));
environment_min_y = min(environment{1}(:,2));
environment_max_y = max(environment{1}(:,2));
X_MIN = floor(environment_min_x-0.1*(environment_max_x-environment_min_x));
X_MAX = floor(environment_max_x+0.1*(environment_max_x-environment_min_x));
Y_MIN = floor(environment_min_y-0.1*(environment_max_y-environment_min_y));
Y_MAX = floor(environment_max_y+0.1*(environment_max_y-environment_min_y));
