function [jointalloc,outlabs,atomsmnsout,time,ipdfout,lambdaout] = BayesMixMultGauss(data , tranvars , compvars , ...
    chains , burn , nrun , thin , K0 , K0lbd , alpha0)
%Summary: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Variable Input descriptions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tranvars: (p2x1) string vector listing continuous variables not 
%            not to be standardized
%           These variables are assumed to marginally be overfitted mixture of normals
% compvars: (kx1) cell of string vectors each listing a unique composition
%          These variables will be ilr transformed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Algorithm Input specifics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% chains: The number of chains to run (recommended 5)
% burn: The number intial sampling iterations to be discarded
% nrun: The total number of iterations to be run
% thin: The thinning interval (record the ith iteration)
% K0: The maximum size of the joint mixture component
% K0lbd: The lower bound of the size of the tensor joint mixture component
% alpha0: The symmetric Dirichlet mixture prior on the tensor joint mixture
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(tranvars); tranvars = string(tranvars); end
if isempty(compvars); compvars = string(compvars); end

% Creating the analysis variables

    ncomps = numel(compvars);
    njcomp = zeros(ncomps , 1);
    compilrs = cell(1,ncomps); 
    complabs = cell(ncomps,1);
    for cc = 1:ncomps
        normcomp = data{:,cellstr(compvars{cc})};
        normcomp = normcomp./sum(normcomp,2);
        compilrs{cc} = ilr(normcomp);
        [~,njcomp(cc)] = size(compilrs{cc});
        complabs{cc} = string(compose('%s_%s' , compvars{cc}(2:end) , compvars{cc}(1)));
    end    
    compilrs = cell2mat(compilrs);
    complabs = vertcat(complabs{:});

    outlabs = [tranvars;complabs];
    yst = [data{:,tranvars},compilrs];

    [n,p0] = size(yst);

%     outtab = table(yst);
%     outtab = splitvars(outtab);
%     outtab.Properties.VariableNames = outlabs;

% The following defines the effective sample size to be output    
    eff_samp1 = (nrun-burn)/thin;

%-- Preallocate placeholders for posterior sampling
    jointalloc = zeros(eff_samp1,n,chains,'uint8');
    lambdaout = zeros(eff_samp1,K0,chains);
    atomsmnsout = zeros(eff_samp1 , p0 , chains);
         

%-- Specifying the initial number of occupied classes each mixture model
% component. Different intialization at each chain. 
    d0 = floor(linspace(K0lbd,K0,chains));
    d0 = d0(randsample(chains,chains));    
    
%-- Specifying base measure hyperparameters
    Sig = (yst - mean(yst))'*(yst - mean(yst));

    nu0 = n+p0;
    kap0 = 1;

    mu0 = mean(yst)';
    Sig0 = 1000*eye(p0);
    
%-- Allocate placeholders for the base measure atom mean and variances
    muInt = zeros(p0 , chains);

%-- Initialize mu
    for jj=1:p0
        muInt(jj,:) = linspace(quantile(yst(:,jj),0.025) , quantile(yst(:,jj),0.975) , chains);
        muInt(jj,:) = muInt(jj,randsample(chains,chains));
    end

    npdf = 100;
    pdfidx = sort(randsample(n , npdf));
    ipdfout = zeros(eff_samp1 , npdf , chains);    
    
%-- Starting Gibbs sampler
tic;
for c = 1:chains
    fprintf('BayesMixMultGauss: Chain %d\n',c);

    % Initilize joint allocation variable
    
    z = zeros(n,1);
    nidx = randsample(n,n);
    nsz = gamrnd(ones(d0(c),1),1);
    nsz = mnrnd(n,nsz./sum(nsz));
    nsz = [0;cumsum(nsz)'];
    for ll = 1:d0(c)
        z(nidx((nsz(ll)+1):nsz(ll+1))) = ll;
    end
        
    %-- Specify the intiial mu values
    mu = muInt(:,c);
    
    for b = 1:nrun    
        %-- Update tensor components
        %-- Update component weights
        [z,~] = find((z == unique(z)')');
        K0 = max(z);
        nz = accumarray(z,1);
        lambda = gamrnd(alpha0 + nz , 1);
        lambda = lambda./sum(lambda);   
        zmat = z == 1:K0;
        
        %-- Update parameters for means and variances     
        pl = kap0./(kap0 + nz);
        ymnl = grpstats(yst,z)';
        mul = mu*pl' + ymnl.*(1-pl)';
        kapl = kap0 + nz;
        nul = nu0 + nz;
        Sigl = zeros(p0,p0,K0);
        
        for ll = 1:K0
            Sigl(:,:,ll) = (yst(zmat(:,ll),:)' - ymnl(:,ll))*(yst(zmat(:,ll),:)' - ymnl(:,ll))';
            Sigl(:,:,ll) = Sig + Sigl(:,:,ll) + (kap0*nz(ll)/kapl(ll))*(ymnl(:,ll)-mu)*(ymnl(:,ll)-mu)';
            Sigl(:,:,ll) = iwishrnd(Sigl(:,:,ll),nul(ll));
            mul(:,ll) = mvnrnd(mul(:,ll) , Sigl(:,:,ll)/kapl(ll));
        end
        
        %-- Update allocations
        cprobs = ones(n,K0);
        for ll = 1:K0
            cprobs(:,ll) = mvnpdf(yst , mul(:,ll)' , Sigl(:,:,ll));
        end
        zupdateprob = lambda'.*cprobs;
        zupdateprob = zupdateprob./sum(zupdateprob,2);
        zupdateprob1 = [zeros(n,1) cumsum(zupdateprob,2)];
        rr = rand(n,1);
        [z,~] = find((rr>=zupdateprob1(:,1:(end-1)) & rr < zupdateprob1(:,2:end))');                    
        ipdf = cprobs(pdfidx, :)*lambda;
        
        %-- Update base measure mean hyper parameters
        Sig0l = zeros(p0);
        for ll = 1:K0
            Sig0l = Sig0l + Sigl(:,:,ll)^-1;
        end
        Sig0l = (Sig0^-1 + Sig0l/kap0)^-1;
        mu0l = zeros(p0,1);
        for ll = 1:K0
            mu0l = mu0l + Sigl(:,:,ll)\mul(:,ll);            
        end
        mu0l = Sig0l*(Sig0\mu0 + mu0l/kap0);
        mu = mvnrnd(mu0l,Sig0l)';
        
        if(numel(unique(z))<2);break;end

        % -- Output samplings -- %
        if (mod(b,thin) == 0 && b > burn)
           lambdaout((b-burn)/thin,1:K0,c) = lambda';
           jointalloc((b-burn)/thin,:,c) = z';
           atomsmnsout((b-burn)/thin,:,c) = mu';           
           ipdfout((b-burn)/thin,:,c) = ipdf';           
        end
        if mod(b,nrun/10) == 0, fprintf('%d%% Complete.\n',b/nrun*100); end
    end   

end
time = toc;

end

