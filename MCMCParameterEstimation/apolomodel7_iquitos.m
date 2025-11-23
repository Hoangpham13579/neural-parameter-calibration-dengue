addpath(genpath('Tool'))

% Declare symbolic funct var
syms  Ms(t) Me(t) Mi(t) Hs(t) He(t) Hi(t) Hr(t) Hitotal(t)

% Declare symbolic var
syms lambda beta_m mu_m theta_m mu_h beta_h theta_h gamma_h

% Model (3) in paper
H=Hs+He+Hi+Hr;
M=Ms+Me+Mi;
ode1 = diff(Ms)  == lambda - beta_m*Hi*Ms/H - (mu_m)*Ms;
ode2 = diff(Me)  == beta_m*Hi*Ms/H - (theta_m+mu_m)*Me;
ode3 = diff(Mi)  == theta_m*Me - mu_m*Mi;
ode4 = diff(Hs)  == mu_h*H - beta_h*Mi/M*Hs - mu_h*Hs;
ode5 = diff(He)  == beta_h*Mi*Hs/M - (theta_h+mu_h)*He;
ode6 = diff(Hi)  == theta_h*He - (gamma_h+mu_h)*Hi;
ode7 = diff(Hr)  == gamma_h*Hi - mu_h*Hr;
ode8 = diff(Hitotal) == theta_h*He;
odes=[ode1; ode2; ode3; ode4; ode5; ode6; ode7; ode8];  % array
vars=[Hitotal Hi Me Hr Hs He Ms Mi];

% ODE solvers: 1st eight vars in the sol vect should be treated as non-negative
opts = odeset('NonNegative',1:8);
load range/Range7_iquitos.mat
% (NOTE) Change [0,48] to correct # of data points when change data
[T,~]=gsua_dpmat(odes,vars,[0 48],'7m','output',1,'opt',opts,'Range',Range, 'solver', 'ode45');

% Construct ydata2 _ cumulative dengue cases
DataIquitos= readtable('data/data_iquitos.csv');
ydata=DataIquitos.CumulativeHumanInfected;
xdata=linspace(0,length(ydata),length(ydata));
ydata2=log1p(ydata);

my = parcluster('local');
delete(my.Jobs);
solver='lsqc';  % least square beltrami solver
p = parpool(4);
tic; % Start timer

% gsua_pe output: 
% % 1. T7 _ summary table with parameter estimation results
% % 2. res _ Vector of Cost funct for each estimation
opt=optimoptions('lsqcurvefit','UseParallel',true,'Display','iter','MaxFunctionEvaluations',4000);
[T7,res]=gsua_pe(T, xdata, ydata2, 'solver', solver, 'N', 1000, 'opt', opt);
save('Result7_iquitos.mat','T7','res','xdata','ydata2')

elapsedTime = toc; % End timer and get elapsed time
disp(['Time to load data: ', num2str(elapsedTime), ' seconds']);