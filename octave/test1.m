% Adapted from https://github.com/pikrakis/Introduction-to-Pattern-Recognition-a-Matlab-Approach/blob/master/Chapter1/examples/example161.m
close('all');
clear;

% "True" parameters for generating the dataset
randn('seed',0);
m1=[1, 1]'; m2=[3, 3]';m3=[2, 6]';
m=[m1 m2 m3];
S(:,:,1)=0.1*eye(2);
S(:,:,2)=0.2*eye(2);
S(:,:,3)=0.3*eye(2);
P=[0.4 0.4 0.2];
N=500;
sed=0;

% Generate the dataset
printf('Generating dataset for Gaussian mixture model\n')
[X,y]=mixt_model(m,S,P,N,sed);

% Save generated data to text files
directory=make_absolute_filename('../test/data/test1');
if isfolder(directory)
  rmdir(directory,'s');
end
mkdir(directory);
printf('Saving test inputs to %s\n', directory)
save('-ascii',[directory, '/observations.txt'],'X')
save('-ascii',[directory, '/classes.txt'],'y')

% Save initial estimates to text files
m1_ini=[0; 2];m2_ini=[5; 2];m3_ini=[5; 5];
m_ini=[m1_ini m2_ini m3_ini];
s_ini=[.15 .27 .4];
Pa_ini=[1/3 1/3 1/3];
e_min=10^(-5);
save('-ascii',[directory,'/initial_mean.txt'],'m_ini')
save('-ascii',[directory,'/initial_covariance.txt'],'s_ini')
save('-ascii',[directory,'/initial_priors.txt'],'Pa_ini')
save('-ascii',[directory,'/error_threshold.txt'],'e_min')
