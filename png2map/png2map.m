clear all;

%%
% Convert images into BW
IM = imread('0.png');
IM = rgb2gray(IM);
BW = imbinarize(IM);

% First find the outer boundary
[OB,OL] = bwboundaries(~BW,'noholes');

% Then the inner hole boundaries
[IB,IL] = bwboundaries(BW,'noholes');

%%

%Robustness constant
epsilon = 0.000000001;
snap_distance = 0.0005;

% TODO: I assume that the second outer boundary is the actual environment 
%       Also assume that the outer boundary is clockwise
%       And inner boundary is counter clockwise
%       These assumptions hold for the specific environment I tested with
%       But might not always be the case. But you can always play around
%       with the flip function used below and the indices to get it to work

environment{1} = [OB{2}(:,2), OB{2}(:,1); OB{2}(1,2) OB{2}(1,1)]; % outer boundary
for k = 2 : length(IB) % holes
    environment = [environment; flip([IB{k}(:,2), IB{k}(:,1); IB{k}(1,2) IB{k}(1,1)]) ];
end

hold on
axis square
plot_environment(environment)

%Select test points with mouse and plot resulting visibility polygon    
while 0 < 1
    
    %Acquire test point.
    [observer_x, observer_y] = ginput(1);
    
    %Make sure the selected point is in the environment
    if ~in_environment( [observer_x observer_y] , environment , epsilon )
        display('Selected points must be in the environment!');
        break;
    end
    
    %Clear plot and form window with desired properties
    cla;

    %Plot environment
    plot_environment(environment)
    
    %Plot observer
    plot3( observer_x , observer_y , 0.3 , ...
           'o' , 'Markersize' , 9 , 'MarkerEdgeColor' , 'y' , 'MarkerFaceColor' , 'k' );
    
    %Compute and plot visibility polygon
    V{1} = visibility_polygon( [observer_x observer_y] , environment , epsilon , snap_distance );
    patch( V{1}(:,1) , V{1}(:,2) , [1, 0, 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    
end

