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
directory=make_absolute_filename('../test/data/test1');
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

% Evaluate Gaussian at observations
[p,n]=size(x);
[J,n]=size(m);
Pa=Pa_ini;
density=[];
denominator=[];
for k=1:p
  tmp=gauss(x(k,:),m,s_ini);
  density=[density;tmp];
  P_tot=tmp*Pa';
  denominator=[denominator, P_tot];
  for j=1:J
    posterior(j,k)=tmp(j)*Pa(j)/P_tot;
  end
end

save('-ascii',[directory,'/densities.txt'],'density')
save('-ascii',[directory,'/denominators.txt'],'denominator')
save('-ascii',[directory,'/posteriors.txt'],'posterior')

% Log likelihood
s=s_ini;
P=posterior;
Q_tot=[];
Q=0;
for k=1:p
    for j=1:J
        Q=Q+P(j,k)*(-(n/2)*log(2*pi*s(j)) - sum( (x(k,:)-m(j,:)).^2)/(2*s(j)) + log(Pa(j)) );
    end
end
Q_tot=[Q_tot Q]

% Update the means
for j=1:J
    a=zeros(1,n);
    for k=1:p
        a=a+P(j,k)*x(k,:);
    end
    m(j,:)=a/sum(P(j,:));
end

save('-ascii',[directory,'/updated_mean.txt'],'m')

% Determine the variances
for j=1:J
    b=0;
    for k=1:p
        b=b+ P(j,k)*((x(k,:)-m(j,:))*(x(k,:)-m(j,:))');
    end
    s(j)=b/(n*sum(P(j,:)));
    
    if(s(j)<10^(-10))
        s(j)=0.001;
    end
end

save('-ascii',[directory,'/updated_covar.txt'],'s')

% Determine the a priori probabilities
for j=1:J
    a=0;
    for k=1:p
        a=a+P(j,k);
    end
    Pa(j)=a/p;
end

save('-ascii',[directory,'/updated_prior.txt'],'Pa')

