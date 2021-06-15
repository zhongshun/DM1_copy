function [Initial_Agent,Initial_Opponent,Initial_Agent_Region,Assets_Collected] = RunDM1(One_Pass,Lookahead,Asset_Position,Negtive_Reward,...
    Negtive_Asset,Number_of_Function,Function_index_size,Visibility_Data,Region,Asset_Visibility_Data,step,Discount_factor,environment,...
    Precompute_Path,WiseUp_Index,heur_penalty_std,heur_agent_detection_weight,heur_agent_asset_weight,epsilon,USE_HEURISTIC)


%% For the heuristic
environment_min_x = min(environment{1}(:,1));
environment_max_x = max(environment{1}(:,1));
environment_min_y = min(environment{1}(:,2));
environment_max_y = max(environment{1}(:,2));
X_MIN = environment_min_x-0.1*(environment_max_x-environment_min_x);
X_MAX = environment_max_x+0.1*(environment_max_x-environment_min_x);
Y_MIN = environment_min_y-0.1*(environment_max_y-environment_min_y);
Y_MAX = environment_max_y+0.1*(environment_max_y-environment_min_y); 

pd = makedist('Normal','mu',0,'sigma',heur_penalty_std);
x = 1:floor(X_MAX+Y_MAX);
PDF_Gaussian = pdf(pd,x);

%% Run one iteration of DM1

for i = 2*Lookahead+1 : -1: 1
    
    % Find all nodes at the current level
    list = find(One_Pass.Nodes.Generation == i);
    
    % If leaf node
    if i == 2*Lookahead+1
        
        % for all leaf nodes
        for j=1:nnz(list)
            
            %   List all the reward value based on the detection of one
            %   of the assests or not
            
            if USE_HEURISTIC
                heuristic_val = Heuristic2(One_Pass.Nodes.Agent{list(j)}, One_Pass.Nodes.Opponent{list(j)}, One_Pass.Nodes.Agent_Region(list(j)),Negtive_Reward,...
                    Negtive_Asset,Asset_Position,One_Pass.Nodes.Assets_Collected{list(j)},environment,Precompute_Path,PDF_Gaussian,ones(size(WiseUp_Index)),heur_agent_detection_weight,heur_agent_asset_weight,Visibility_Data);
            else
                heuristic_val = 0;
            end
                        
            % minimal evaluation function
            E_them =  One_Pass.Nodes.Current_Step_reward_wo_assets(list(j)) + heuristic_val;
 
            E_them_temp = E_them;
            
            Assets_Collected = One_Pass.Nodes.Assets_Collected{list(j)}; %indicator to label which asset is collected along the path to this node
            
            for Function_M = 0:Number_of_Function-1
                E_them = E_them_temp;
                Index = Function_M;
                for N = Function_index_size:-1:1
                    if  mod(Index,2)
                        E_them = E_them - (Discount_factor^Assets_Collected(N))...
                            * (Assets_Collected(N)>0) * (Negtive_Asset);
                    end
                    Index = floor(Index/2);
                end
                One_Pass_Nodes_E_them(1,Function_M+1) = E_them;
            end
            
            
            One_Pass.Nodes.E_them{list(j)} = One_Pass_Nodes_E_them;
            One_Pass.Nodes.E_us(list(j)) = E_them;
            One_Pass.Nodes.Decision_Node(list(j)) = list(j);
            
        end
    elseif ~mod(i,2) %MIN Level
        for j = 1:nnz(list)
            %             Children_node = successors(One_Pass,list(j));
            
  
            Children_node = One_Pass.Nodes.Successors{list(j)};
            
            %Find which function we need to use based on the wise up state
            %of the opponent
            Decision_Index_E_them = 0;
            M = length(One_Pass.Nodes.Assets_Detected{list(j)});
            for CheckBit = 1:M
                if One_Pass.Nodes.Assets_Detected{list(j)}(CheckBit) == 1
                    Decision_Index_E_them = bitset(Decision_Index_E_them,M - CheckBit + 1);
                end
            end
            %             Decision_Index_E_them = bin2dec(num2str(One_Pass.Nodes.Assets_Detected{list(j)}'))+1;
            Decision_Index_E_them = Decision_Index_E_them + 1;
            
            
            
            Best_value = One_Pass.Nodes.E_them{Children_node(1)}(Decision_Index_E_them);
            
            for k = 1:nnz(Children_node)
                if Best_value > One_Pass.Nodes.E_them{Children_node(k)}(Decision_Index_E_them)
                    Best_value = One_Pass.Nodes.E_them{Children_node(k)}(Decision_Index_E_them);
                end
            end
            
            %Find the minimal value based on the wise up state of the
            %opponent
            Best_nodes = [];
            P = list(j);
            for k = 1:nnz(Children_node)
                if One_Pass.Nodes.E_them{Children_node(k)}(Decision_Index_E_them) == Best_value
                    Best_nodes(length(Best_nodes) + 1) = Children_node(k);
                end
            end
            
            
            % get the minimal reward for current step
            Best_node = Best_nodes(1);
            P = list(j);
            
            %The Opponent chose to stay if staying is among one of the
            %optimal actions of its child nodes.
            
            if nnz(Best_nodes)> 1
                a = 1;
            end
            for k = 1:nnz(Best_nodes)
                if One_Pass.Nodes.Opponent{P}(1) == One_Pass.Nodes.Opponent{Best_nodes(k)}(1) &&...
                        One_Pass.Nodes.Opponent{P}(2) == One_Pass.Nodes.Opponent{Best_nodes(k)}(2)
                    Best_node = Best_nodes(k);
                    break
                end
            end


            %Test the effect of moving toward the agent when tie-breaking,
            %because chose the appropriate heuristic weight for the opponent detection
            %part is not very straightforward, could make some strange
%             %behaviors for the agent or the opponent.
% 
%             for k = 1:nnz(Best_nodes)
%                 if norm(One_Pass.Nodes.Opponent{Best_nodes(k)}-One_Pass.Nodes.Agent{Best_nodes(k)}) < ...
%                         norm(One_Pass.Nodes.Opponent{Best_node}-One_Pass.Nodes.Agent{Best_node})
%                     Best_node = Best_nodes(k);
% %                     break
%                 end
%             end
            
            
            
            One_Pass.Nodes.Decision_Value(list(j)) = One_Pass.Nodes.E_them{Best_node}(Decision_Index_E_them);
            One_Pass.Nodes.Decision_Node(list(j)) = Best_node;
            One_Pass.Nodes.E_us(list(j)) = One_Pass.Nodes.E_us(Best_node);
            One_Pass.Nodes.E_them(list(j)) = One_Pass.Nodes.E_them(Best_node);
        end
    else %MAX Level
        for j = 1:nnz(list)
            %             Children_node = successors(One_Pass,list(j));
            Children_node = One_Pass.Nodes.Successors{list(j)};

            
            
            Best_value = One_Pass.Nodes.E_us(Children_node(1));
            
            for k = 1:nnz(Children_node)
                if Best_value < One_Pass.Nodes.E_us(Children_node(k))
                    Best_value = One_Pass.Nodes.E_us(Children_node(k));
                end
            end
            
            %Find the maximal value based on the wise up state of the
            %opponent
            Best_nodes = [];
            for k = 1:nnz(Children_node)
                if One_Pass.Nodes.E_us(Children_node(k)) == Best_value
                    Best_nodes(nnz(Best_nodes) + 1) = Children_node(k);
                end
            end
            Best_node = Best_nodes(1);
            P = list(j);

            
            %The agent chose to stay if staying is among one of the
            %optimal actions of its child nodes.
            
            for k = 1:nnz(Best_nodes)
                if One_Pass.Nodes.Agent{P}(1) == One_Pass.Nodes.Agent{Best_nodes(k)}(1) &&...
                        One_Pass.Nodes.Agent{P}(2) == One_Pass.Nodes.Agent{Best_nodes(k)}(2)
                    Best_node = Best_nodes(k);
                    break
                end
            end

            %Test the effect of moving away from the opponent when tie-breaking,
            %because chose the appropriate heuristic weight for the opponent detection
            %part is not very straightforward, could make some strange
            %behaviors for the agent or the opponent.
%             for k = 1:nnz(Best_nodes)
%                 if norm(One_Pass.Nodes.Opponent{Best_nodes(k)}-One_Pass.Nodes.Agent{Best_nodes(k)}) > ...
%                         norm(One_Pass.Nodes.Opponent{Best_node}-One_Pass.Nodes.Agent{Best_node})
%                     Best_node = Best_nodes(k);
% %                     break
%                 end
%             end


            
            One_Pass.Nodes.Decision_Node(list(j)) = Best_node;
            One_Pass.Nodes.E_us(list(j)) = One_Pass.Nodes.E_us(Best_node);
            One_Pass.Nodes.Decision_Value(list(j)) = One_Pass.Nodes.E_them{Best_node}(Decision_Index_E_them);
            
            %Update E_them
            %            E_them = One_Pass.Nodes.E_them{Children_node(1)};
            %            for M = 1:Number_of_Function
            %                 for k = 1:nnz(Children_node)
            %                     One_Pass_Nodes_E_them_temp = One_Pass.Nodes.E_them{Children_node(k)};
            %                     E_them(M) = max(E_them(M),One_Pass_Nodes_E_them_temp(M));
            %                 end
            %            end
            
            E0 = One_Pass.Nodes.E_them{Children_node(1)};
            
            for k = 1:nnz(Children_node)
                One_Pass_Nodes_E_them_temp = One_Pass.Nodes.E_them{Children_node(k)};
                for M = 1:Number_of_Function
                    E0(M) = max(E0(M),One_Pass_Nodes_E_them_temp(M));
                end
            end
            
            One_Pass.Nodes.E_them{list(j)} = E0;
            
        end
    end
    
end



% %find the optimal path
One_Pass_Node_path = 1;
One_Pass_Best_node = 1;
for i = 2:2*Lookahead+1
    One_Pass_Best_node = One_Pass.Nodes.Decision_Node(One_Pass_Best_node);
    One_Pass_Node_path = [One_Pass_Node_path One_Pass_Best_node];
end

% %find the optimal path

for k =1:2:nnz(One_Pass_Node_path)
    Agent_path_x((k+1)/2) = One_Pass.Nodes.Agent{One_Pass_Node_path(k)}(1);
    Agent_path_y((k+1)/2) = One_Pass.Nodes.Agent{One_Pass_Node_path(k)}(2);
    
    Opponent_path_x((k+1)/2) = One_Pass.Nodes.Opponent{One_Pass_Node_path(k)}(1);
    Opponent_path_y((k+1)/2) = One_Pass.Nodes.Opponent{One_Pass_Node_path(k)}(2);
end


Initial_Agent_Region = One_Pass.Nodes.Agent_Region{One_Pass_Node_path(3)};
Initial_Agent = [Agent_path_x(2);Agent_path_y(2)];
Initial_Opponent = [Opponent_path_x(2);Opponent_path_y(2)];

Assets_Collected = One_Pass.Nodes.Assets_Collected{One_Pass_Node_path(3)};
save('Save_Visibility_Data\Show_Tree.mat');
Plot_Path_Online_DM1;
end