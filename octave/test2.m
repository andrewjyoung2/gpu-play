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

% Transpose the observations and means so that each row is a vector
x=X';     % 500 x 2

% Save generated data to text files
directory=make_absolute_filename('../test/data/test2');
if isfolder(directory)
  rmdir(directory,'s');
end
mkdir(directory);
printf('Saving test inputs to %s\n', directory)
save('-ascii',[directory, '/observations.txt'],'x')
save('-ascii',[directory, '/classes.txt'],'y')

% Save initial estimates to text files
m1_ini=[0; 2];m2_ini=[5; 2];m3_ini=[5; 5];
m_ini=[m1_ini m2_ini m3_ini];
s_ini=[.15 .27 .4];
Pa_ini=[1/3 1/3 1/3];
e_min=10^(-5);

m=m_ini'; % 3 x 2
save('-ascii',[directory,'/initial_mean.txt'],'m')
save('-ascii',[directory,'/initial_covariance.txt'],'s_ini')
save('-ascii',[directory,'/initial_priors.txt'],'Pa_ini')
save('-ascii',[directory,'/error_threshold.txt'],'e_min')

[m_hat,s_hat,Pa,iter,Q_tot,e_tot]=em_alg_function(X,m_ini,s_ini,Pa_ini,e_min);

save('-ascii',[directory,'/mean_est.txt'],'m_hat')
save('-ascii',[directory,'/covar_est.txt'],'s_hat')
save('-ascii',[directory,'/prior_est.txt'],'Pa')
save('-ascii',[directory,'/num_iter.txt'],'iter')
save('-ascii',[directory,'/error_est.txt'],'e_tot')

