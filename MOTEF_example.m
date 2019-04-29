%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OBJECTIVE: The following program was created to illustrate the 
%            use of the MOTEF function.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PROGRAMMER: Angel Davalos (adjdavalols@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% README:
% 
% The code below requires the MOTEF function to be loaded by adding a path to
% the location where the MOTEF.m file is stored. This code runs using
% MATLAB2018b. Files needed to be stored are listed below:
%
%  1. MOTEF.m
%  2. trandn.m 
%  3. pnorm2_mex.mexw64 (if using LINUX pnorm2_mex.mexw64)
%  4. minBinderLoop2_mex.mexw64 (if using LINUX minBinderLoop2_mex.mexa64)
%  5. mpsrf.m 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

% clear;clc;

basefldr = '~'; % Specify location where Functions folder is stored

% The following loads MOTEF functions
addpath(sprintf('%s%s',basefldr,'/Functions'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read in data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

infile = sprintf('%s%s',basefldr,"/Data/MOTEF_example.mat");
load(infile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-- Specifying inputs for MOTEF function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Specifying string vectors that correspond to variable names in the data
% table by varible scale type.
catvars = string(compose('cat_%02.0f',1:14))';
convars = string(compose('con_%02.0f',1:3))';
couvars = string(compose('cou_%02.0f',1:3))';

% Specifying chain inputs
chains = 5;
burn = 1000;
nrun = 5000;
thin = 4;
K0 = 20; K0lbd = 10;
K = 10; Klbd = 5;
[alpha0,alpha] = deal(1e-25);
nsamp = size(data,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-- Running MOTEF Gibbs sampler
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(1342);
[jointalloc,labelsout,margalloc,~,time,infout,...
    ~] = MOTEF(data,convars,couvars,[],[],catvars,...
    chains,burn,nrun,thin,K0,K0lbd,K,Klbd,alpha0,alpha,true);

%-- Computing point estimate of the joint clustering variables by
%-- minimizing Binder's loss

jointalloctmp = reshape(permute(jointalloc,[1,3,2]),[],nsamp);
tic;
[optclu,optitr] = minBinderLoop2_mex(jointalloctmp);
time2 = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-- Examining some output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The following summarizes the selected optimal joint clustering
tabulate(optclu)

% The following performs a cross tabulation between the true clustering
% (joint_clu_t) and the model-based optimal clustering (optclu)
crosstab(joint_clu_t,optclu)

% The following computes the multivariate potential scale reduction factor 
% for assessing convergence of all the paiwise measure of mutual
% information for assessing pairwise dependence among all pairs of
% variables.
mpsrf(infout)

% The following is a summary plot for assessing pairwise dependence among
% all pairs of variables
p = numel(labelsout);
[pinfest,tdm] = deal(zeros(p,p));

trudep = [catvars([2,3,7,8,12]);convars(1);couvars(1)];

epostpinf = mean(mean(infout > 0),3)';
contl = 0;
for j1 = 1:(p-1)
    for j2 = (j1+1):p
        contl = contl + 1;
        [pinfest(j1,j2),pinfest(j2,j1)] = deal(epostpinf(contl));
        [tdm(j1,j2),tdm(j2,j1)] = deal(prod(ismember([labelsout(j1),...
                                    labelsout(j2)],trudep)));
    end
end


figure('Units' , 'inches' , 'Position', [0, 0, 22.5, 4.2]);
subplot(1,3,1);heatmap(labelsout,...
    labelsout,tdm,'Colormap',flipud(hot));
title("True Dependence");
subplot(1,3,2);heatmap(labelsout,...
    labelsout,pinfest,'Colormap',flipud(hot));
title("Posterior Probability of Dependence");
subplot(1,3,3);heatmap(labelsout,...
labelsout,1*(pinfest>0.95),'Colormap',flipud(hot));
title("Posterior Probability of Dependence > 0.95");

saveas(gcf , sprintf('%s%s',basefldr,"/MOTEF_example_plots.jpg"));
close(gcf);

% The following functions were created by other programmers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: mpsrf.m code
% Author: Simo Sarkka & Aki Vehtari
% Date: 1999 - 2004
% Code version: 1.0
% https://github.com/translationalneuromodeling/tapas/tree/master/mpdcm/external/mcmcdiag
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
