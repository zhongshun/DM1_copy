function Reward_visbility = Heuristic2_minimax(Initial_Robot, Initial_Opponent, Agent_Region,Negtive_Reward,...
    Negtive_Asset,Assets,Assets_Collect,environment,Precompute_Path,PDF,WiseUp_Index)


% pd = makedist('Normal','mu',0,'sigma',9);
% x = 1:20;
% PDF = pdf(pd,x);
% 
% environment = read_vertices_from_file('./Environments/M_starstar1.environment');
environment_min_x = min(environment{1}(:,1));
environment_max_x = max(environment{1}(:,1));
environment_min_y = min(environment{1}(:,2));
environment_max_y = max(environment{1}(:,2));
X_MIN = environment_min_x-0.1*(environment_max_x-environment_min_x);
X_MAX = environment_max_x+0.1*(environment_max_x-environment_min_x);
Y_MIN = environment_min_y-0.1*(environment_max_y-environment_min_y);
Y_MAX = environment_max_y+0.1*(environment_max_y-environment_min_y);
epsilon = 0.0001;


% pd = makedist('Normal','mu',0,'sigma',9);
% x = 1:X_MAX;
% PDF = pdf(pd,x);

X_max = floor(X_MAX);
Y_max = floor(Y_MAX);

start_postion = Initial_Robot;
k = 1;
Postions = start_postion;

%Compute the positive reward
Reward_visbility = 0;
for x = max(floor(X_MIN),1):floor(X_MAX)+1
    for y = max(floor(Y_MIN),1):floor(Y_MAX)+1
        if in_environment( [x,y] , environment , epsilon ) && ~ Agent_Region{1}(x,y)
            D = Precompute_Path{X_max*Postions(2)+Postions(1),X_max*y+x};
            Reward_visbility = Reward_visbility +  PDF(length(D(1,:)));
        end
    end
end

%Compute the negative reward of the opponent

Penalty_Opponent = 0;

D = Precompute_Path{X_max*Postions(2,k)+Postions(1,k),X_max*Initial_Opponent(2)+Initial_Opponent(1)};
Penalty_Opponent = Penalty_Opponent +  150*Negtive_Reward*PDF(length(D(1,:)));


%Compute the negative reward of the asset
Penalty_Asset = 0;
for k = 1:length(Postions(1,:))
    for L = 1:length(Assets(1,:))
        if Assets_Collect(L) == 1 || WiseUp_Index(L) == 0
            continue;
        end
        D = Precompute_Path{X_max*Postions(2,k)+Postions(1,k),X_max*Assets(L,2)+Assets(L,1)};
        Penalty_Asset = Penalty_Asset +  Negtive_Asset* PDF(length(D(1,:)));
    end
end

Reward_visbility = Reward_visbility - Penalty_Opponent - Penalty_Asset;

end