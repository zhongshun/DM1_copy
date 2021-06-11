function plot_environment(environment)

plot(environment{1}(:,1), environment{1}(:,2),'k','LineWidth',2)
for k = 2 : length(environment)
    fill(environment{k}(:,1), environment{k}(:,2),[.5, .5, .5])
end