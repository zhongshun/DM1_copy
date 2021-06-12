
%Read environment geometry from file
% environment = read_vertices_from_file('./Environments/M_starstar12.environment');
environment_min_x = min(environment{1}(:,1));
environment_max_x = max(environment{1}(:,1));
environment_min_y = min(environment{1}(:,2));
environment_max_y = max(environment{1}(:,2));
X_MIN = floor(environment_min_x-0.1*(environment_max_x-environment_min_x));
X_MAX = floor(environment_max_x+0.1*(environment_max_x-environment_min_x));
Y_MIN = floor(environment_min_y-0.1*(environment_max_y-environment_min_y));
Y_MAX = floor(environment_max_y+0.1*(environment_max_y-environment_min_y));

ENV_SIZE1 = floor(X_MAX)+1;  % will be ENV_SIZE x ENV_SIZE grid
ENV_SIZE2 = floor(Y_MAX)+1;

for x = floor(X_MIN)+1:floor(X_MAX)+1
    for y = floor(Y_MIN)+1:floor(Y_MAX)+1
        if in_environment( [x,y] , environment , epsilon )
            Visibility_Data{X_MAX*y + x} = visibility_polygon( [x y] , environment , epsilon, snap_distance); 
            Region{X_MAX*y + x} =  poly2mask(Resolution*Visibility_Data{X_MAX*y + x}(:,1),Resolution*Visibility_Data{X_MAX*y + x}(:,2),Resolution*ENV_SIZE1, Resolution*ENV_SIZE2);
        else
            if X_MAX*y + x <= 0
                continue
            end 
            Visibility_Data{X_MAX*y + x} = -1; 
        end
    end
end

for i = 1:length(Visibility_Data)
    Visibility_Data{i} = -1;
end

for i = 1:length(Region)
    Region{i} = -1;
end

for x = floor(X_MIN)+1:floor(X_MAX)+1
    for y = floor(Y_MIN)+1:floor(Y_MAX)+1
        if in_environment( [x,y] , environment , epsilon )
            Visibility_Data{X_MAX*y + x} = visibility_polygon( [x y] , environment , epsilon, snap_distance); 
            Region{X_MAX*y + x} =  poly2mask(Resolution*Visibility_Data{X_MAX*y + x}(:,1),Resolution*Visibility_Data{X_MAX*y + x}(:,2),Resolution*ENV_SIZE1, Resolution*ENV_SIZE2);
        else
            if X_MAX*y + x <= 0
                continue
            end
            Visibility_Data{X_MAX*y + x} = -1;
        end
    end
end



for i = 1:length(Asset(:,1))
    w{i} =  visibility_polygon( Asset(i,:) , environment , epsilon, snap_distance);
    for x = floor(X_MIN)+1:floor(X_MAX)+1
        for y = floor(Y_MIN)+1:floor(Y_MAX)+1
            if X_MAX*y + x <= 0
                continue
            end
            if in_environment( [x,y] , w , epsilon )
                Asset_Visibility_Data(i,X_MAX*y + x) = 1;
            else
                Asset_Visibility_Data(i,X_MAX*y + x) = -1;
            end
        end
    end
end

Visibility_in_environment = zeros(floor(X_MAX)+1 + X_MAX* (floor(Y_MAX)+1), floor(X_MAX)+1 + X_MAX* (floor(Y_MAX)+1));

for x_location = floor(X_MIN)+1:floor(X_MAX)+1
    for y_location = floor(Y_MIN)+1:floor(Y_MAX)+1
        if in_environment( [x_location,y_location] , environment , epsilon )
            V{1} =  visibility_polygon( [x_location,y_location] , environment , epsilon, snap_distance);
%             display([x_location,y_location]);
            for x_tosee = floor(X_MIN)+1:floor(X_MAX)+1
                for y_tosee = floor(Y_MIN)+1:floor(Y_MAX)+1             
                    if in_environment( [x_tosee,y_tosee] , V , epsilon )
                        x = x_location + X_MAX*y_location;
                        y = x_tosee + X_MAX* y_tosee;
                        Visibility_in_environment(x,y) = 1;
                    end             
                end
            end
        end
        
    end
end

%%Creat_Environemt_Graph, to record the shorest distance from one node to
%%another node

G = graph([], []);
G = addnode(G,(floor(X_MAX)+1)*(floor(Y_MAX)+1));
X_max = floor(X_MAX);
Y_max = floor(Y_MAX);

for x = floor(X_MIN):floor(X_MAX)+1
    for y = floor(Y_MIN):floor(Y_MAX)+1

        if in_environment( [x,y] , environment , epsilon )
            if in_environment( [x-1,y] , environment , epsilon )
                G = addedge(G,X_max*y+x,X_max*y+(x-1));
            end
            
            if in_environment( [x+1,y] , environment , epsilon )
                G = addedge(G,X_max*y+x,X_max*y+(x+1));
            end
            
             if in_environment( [x,y-1] , environment , epsilon )
                G = addedge(G,X_max*y+x,X_max*(y-1)+x);
             end
            
             if in_environment( [x,y+1] , environment , epsilon )
                G = addedge(G,X_max*y+x,X_max*(y+1)+x);
             end
        end
    end
end

for i = 1:X_max*Y_max
  x_i = mod(i,X_max);
  y_i = (i - x_i)/X_max;
  
  if ~in_environment( [x_i,y_i] , environment , epsilon )
     continue; 
  end
  
  for j = 1:X_max*Y_max
      x_j = mod(i,X_max);
      y_j = (i - x_j)/X_max;
      if ~in_environment( [x_i,y_i] , environment , epsilon )
          continue;
      end
      
      P = shortestpath(G,i,j);
      path = [];
      for n = 1:length(P)
          path(1,n) = mod(P(n),X_max);
          path(2,n) = floor(P(n)/X_max);
      end
      
      Precompute_Path{i,j} = path;
      
  end
  
  
end



save('Save_Visibility_Data\M_starstar12.mat')