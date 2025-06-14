function [X,y]=mixt_model(m,S,P,N,sed)
% Adapted from https://github.com/pikrakis/Introduction-to-Pattern-Recognition-a-Matlab-Approach/blob/master/Chapter1/mfiles/mixt_model.m

% Load statistics package for mvnrnd
pkg load statistics

rand('seed',sed);
[l,c]=size(m);

% Accumulate the vector of priors
P_acc=P(1);
for i=2:c
  t=P_acc(i-1)+P(i);
  P_acc=[P_acc t];
end

% Generate dataset
X=[];
y=[];
for i=1:N
  t=rand;
  idx=sum(t>P_acc)+1;
  X=[X; mvnrnd(m(:,idx)',S(:,:,idx),1)];
  y=[y idx];
end
X=X';
