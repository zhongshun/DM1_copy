epsilon = 0.000000001;  %Robustness constant
snap_distance = 0.05;   %Snap distance (distance within which an observer location will be snapped to the
                        %boundary before the visibility polygon is computed)


Resolution = 1;         % 1 is a coarse grid used for computing visibility

Negtive_Reward = 0.3;   % penalty for agent being detected
Negtive_Asset = 30;     % penalty for asset being detected

Lookahead = 5;          % planning horizon        

T_execution = 10;       % episode duration
Discount_factor = 0.95;

heur_penalty_std = 9;        % std of penalty to use for the heuristic
heur_agent_detection_weight = 50;   % how much weight to give the opponent detecting the agent penalty in the heuristic
