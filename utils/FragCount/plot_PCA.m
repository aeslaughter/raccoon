close all
clc

filename = 'correlation_length_5';
num_trials = 5;

V = [];
for i = 1:num_trials
  V = [V;dlmread([filename,'_trial_',num2str(i),'.e.PCA.csv'],',')];
end

VV = [V;-V];

theta = [];

for i = 1:size(V,1)

  v = V(i,:);
  theta = [theta,atan(v(2)/v(1))+pi/2];
  theta = [theta,theta(end)+pi];

end

% figure
% plot(VV(:,1),VV(:,2),'.')
% axis equal


Theta = [theta-2*pi,theta,theta+2*pi];
[f,xi] = ksdensity(Theta,'Function','pdf','Support',[-2*pi,4*pi],'NumPoints',300,'BandWidth',0.02);
% figure
% plot(xi,f)

subplot(1,3,1)
polarplot(xi(100:200),f(100:200),'-')
subplot(1,3,2)
polarhistogram(theta,100,'Normalization','probability','DisplayStyle','stairs');
subplot(1,3,3)
polarhistogram(theta,40,'EdgeColor','none');