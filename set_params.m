epsilon = 0.000000001;  %Robustness constant
snap_distance = 0.05;   %Snap distance (distance within which an observer location will be snapped to the
                        %boundary before the visibility polygon is computed)


Resolution = 1;         % 1 is a coarse grid used for computing visibility

Negtive_Reward = 0.3;   % penalty for agent being detected
Negtive_Asset = 30;     % penalty for asset being detected

Lookahead = 3;          % planning horizon        

T_execution = 10;       % episode duration
Discount_factor = 0.93;

heur_penalty_std = 1;        % std of penalty to use for the heuristic
heur_agent_detection_weight = 0.01;   % how much weight to give the opponent detecting the agent penalty in the heuristic

heur_agent_explore_weight = 2;   % how much weight to give the opponent detecting the agent penalty in the heuristic