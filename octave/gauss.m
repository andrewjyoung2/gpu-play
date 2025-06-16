function [z]=gauss(x,m,s)

% https://github.com/pikrakis/Introduction-to-Pattern-Recognition-a-Matlab-Approach/blob/master/Chapter1/mfiles/gauss.m

[J,l]=size(m);
[p,l]=size(x);
z=[];
for j=1:J
    t=(x-m(j,:))*(x-m(j,:))';
    c=1/(2*pi*s(j))^(l/2);
    z=[z c*exp(-t/(2*s(j)))];
end
