function [Initial_Agent,Initial_Opponent,Initial_Agent_Region,Assets_Collected] = RunMinimax(Minimax_Pass,T,Asset_Position,Negtive_Reward,...
    Negtive_Asset,Number_of_Function,Function_index_size,Visibility_Data,Region,Asset_Visibility_Data,Visibility_in_environmentstep,step,Discount_factor,...
    environment,Precompute_Path,WiseUp_Index)

% RunDM1(Tree,T,Asset,Negtive_Reward,Negtive_Asset,Number_of_Function,Function_index_size,
% Visibility_Data,Region,Asset_Visibility_Data,step)

epsilon = 0.0001;
% Discount_factor = 1;
pd = makedist('Normal','mu',0,'sigma',9);
x = 1:50;
% PDF_Gaussian = pdf(pd,x);
PDF_Gaussian = 1./(10+x);

for i = 2*T+1 :-1:1
    list =  find(Minimax_Pass.Nodes.Generation == i);
    if i == 1
       a = 1; 
    end
    if i == 2*T+1
        for j=1:nnz(list)
            %   List all the reward value based on the detection of one
            %   of the assests or not
            Reward_visbility = Heuristic2_minimax(Minimax_Pass.Nodes.Agent{list(j)}, Minimax_Pass.Nodes.Opponent{list(j)}, Minimax_Pass.Nodes.Agent_Region(list(j)),Negtive_Reward,...
                Negtive_Asset,Asset_Position,Minimax_Pass.Nodes.Detection_Asset_Collect{list(j)},environment,Precompute_Path,PDF_Gaussian,WiseUp_Index);
            
            E_them =  Minimax_Pass.Nodes.Current_Step_reward_wo_assets(list(j)) + Reward_visbility;
            Detection_Asset_Collect = Minimax_Pass.Nodes.Detection_Asset_Collect{list(j)};
            for N = Function_index_size:-1:1
                E_them = E_them - (Discount_factor^Detection_Asset_Collect(N))...
                            * (Detection_Asset_Collect(N)>0) * Negtive_Asset;
            end
            
            Minimax_Pass.Nodes.Best_value(list(j)) = E_them;
            Minimax_Pass.Nodes.Decision_Node(list(j)) = list(j);
            
        end
    elseif mod(i,2) %Min Level since opponent moves first
        for j = 1:nnz(list)
            Children_node = Minimax_Pass.Nodes.Successors{list(j)};
            %Find which function we need to use based on the wise up state
            %of the opponent
  
            
            Best_value = Minimax_Pass.Nodes.Best_value(Children_node(1));
            
            for k = 1:nnz(Children_node)
                Best_value = min(Best_value,Minimax_Pass.Nodes.Best_value(Children_node(k)));
            end
            
            %Find the minimal value based on the wise up state of the
            %opponent
            Best_nodes = [];
            P = list(j);
            for k = 1:nnz(Children_node)
                if Minimax_Pass.Nodes.Best_value(Children_node(k)) == Best_value
                    Best_nodes(length(Best_nodes) + 1) = Children_node(k);
                end
            end
            
            
%             Best_node = Best_nodes(1);
            % get the minimal reward for current step
%             if length(Best_nodes) > 1
%                 Best_one = 1;
%                 for B = 1:nnz(Best_nodes)
%                     
%                     if Minimax_Pass.Nodes.Opponent{Best_nodes(B)}(1) == Minimax_Pass.Nodes.Opponent{P}(1) &&...
%                             Minimax_Pass.Nodes.Opponent{Best_nodes(B)}(1) == Minimax_Pass.Nodes.Opponent{P}(2)
%                         Best_one = B;
%                         break;
%                         %                                 elseif Minimax_Pass.Nodes.Current_Step_reward_with_assest(Best_nodes(B)) < Minimax_Pass.Nodes.Current_Step_reward_with_assest(Best_nodes(Best_one))
%                         %                                     Best_one = B;
%                     end
%                 end
%                 Best_node = Best_nodes(Best_one);

                
                Best_node = Best_nodes(1);
                P = list(j);
                
                
                for k = 1:nnz(Best_nodes)
                    if Minimax_Pass.Nodes.Opponent{P}(1) == Minimax_Pass.Nodes.Opponent{Best_nodes(k)}(1) &&...
                            Minimax_Pass.Nodes.Opponent{P}(2) == Minimax_Pass.Nodes.Opponent{Best_nodes(k)}(2)
                        Best_node = Best_nodes(k);
                        break
                    end
                end
                
            
            %             Minimax_Pass.Nodes.Decision_Value(list(j)) = Minimax_Pass.Nodes.E_them{Best_node}(Decision_Index);
            Minimax_Pass.Nodes.Decision_Node(list(j)) = Best_node;
            Minimax_Pass.Nodes.Best_value(list(j)) = Minimax_Pass.Nodes.Best_value(Best_node);
%             Minimax_Pass.Nodes.E_them(list(j)) = Minimax_Pass.Nodes.E_them(Best_node);
        end
    else %MAX Level
        for j = 1:nnz(list)
            %             Children_node = successors(One_Pass,list(j));

            Children_node = Minimax_Pass.Nodes.Successors{list(j)};
            
            Best_value = Minimax_Pass.Nodes.Best_value(Children_node(1));
            
            for k = 1:nnz(Children_node)
                Best_value = max(Best_value,Minimax_Pass.Nodes.Best_value(Children_node(k)));
            end
            
            Minimax_Pass.Nodes.Best_value(list(j)) = Best_value;
            
            %Find the maximal value based on the wise up state of the
            %opponent
            Best_nodes = [];
            for k = 1:nnz(Children_node)
                if Minimax_Pass.Nodes.Best_value(Children_node(k)) == Best_value
                    Best_nodes(nnz(Best_nodes) + 1) = Children_node(k);
                end
            end
            
            Best_node = Best_nodes(1);
            P = list(j);

                for k = 1:nnz(Best_nodes)
                    if Minimax_Pass.Nodes.Opponent{P}(1) == Minimax_Pass.Nodes.Opponent{Best_nodes(k)}(1) &&...
                            Minimax_Pass.Nodes.Opponent{P}(2) == Minimax_Pass.Nodes.Opponent{Best_nodes(k)}(2)
                        Best_node = Best_nodes(k);
                        break
                    end
                end
  
            
            Minimax_Pass.Nodes.Decision_Node(list(j)) = Best_node;
            Minimax_Pass.Nodes.Best_value(list(j)) = Minimax_Pass.Nodes.Best_value(Best_node);
%             Minimax_Pass.Nodes.Decision_Value(list(j)) = Minimax_Pass.Nodes.Best_value{Best_node};
            

            
            
        end
    end
    
end



% %find the optimal path
One_Pass_Node_path = 1;
One_Pass_Best_node = 1;
for i = 2:2*T+1
    One_Pass_Best_node = Minimax_Pass.Nodes.Decision_Node(One_Pass_Best_node);
    One_Pass_Node_path = [One_Pass_Node_path One_Pass_Best_node];
end

% %find the optimal path

for k =1:2:nnz(One_Pass_Node_path)
    
    Agent_path_x((k+1)/2) = Minimax_Pass.Nodes.Agent{One_Pass_Node_path(k)}(1);
    Agent_path_y((k+1)/2) = Minimax_Pass.Nodes.Agent{One_Pass_Node_path(k)}(2);
    
    Opponent_path_x((k+1)/2) = Minimax_Pass.Nodes.Opponent{One_Pass_Node_path(k)}(1);
    Opponent_path_y((k+1)/2) = Minimax_Pass.Nodes.Opponent{One_Pass_Node_path(k)}(2);
end

Initial_Agent = [Agent_path_x(2);Agent_path_y(2)];
% for k = 2:2:length(One_Pass_Node_path)
%     if Initial_Agent(1) ~= Minimax_Pass.Nodes.Agent{One_Pass_Node_path(k)}(1) || Initial_Agent(2) ~= Minimax_Pass.Nodes.Agent{One_Pass_Node_path(k)}(2)
%         Initial_Agent =  Minimax_Pass.Nodes.Agent{One_Pass_Node_path(k)};
%         break;
%     end
% end
Initial_Agent_Region = Minimax_Pass.Nodes.Agent_Region{One_Pass_Node_path(3)};


Initial_Opponent = [Opponent_path_x(2);Opponent_path_y(2)];
% for k = 3:2:length(One_Pass_Node_path)
%     if Initial_Opponent(1) ~= Minimax_Pass.Nodes.Opponent{One_Pass_Node_path(k)}(1)  || Minimax_Pass.Nodes.Opponent{One_Pass_Node_path(k)}(2)
%         Initial_Opponent =  Minimax_Pass.Nodes.Opponent{One_Pass_Node_path(k)};
%         break;
%     end
% end


Assets_Collected = Minimax_Pass.Nodes.Detection_Asset_Collect{One_Pass_Node_path(3)};


save('Save_Visibility_Data\Show_Tree.mat');
Plot_Path_Online;
end