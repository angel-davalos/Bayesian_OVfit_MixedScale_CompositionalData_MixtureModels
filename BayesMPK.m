function [jointalloc,outlabs,atomsmnsout,time,infout,ipdfout,lambdaout] = BayesMPK(data , convars , tranvars , compvars , ...
    couvars , catvars , chains , burn , nrun , thin , K0 , K0lbd , ...
    alpha0 , MIout)
%Summary: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Variable Input descriptions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% convars: (p1x1) string vector listing continuous variables 
% tranvars: (p2x1) string vector listing continuous variables not 
%            not to be standardized
%           These variables are assumed to marginally be overfitted mixture of normals
% compvars: (kx1) cell of string vectors each listing a unique composition
%          These variables will be ilr transformed
%  convars,tranvars,compars are all assumed to have Gaussian kernels
% couvars: (p3x1) string vector listing count/ordinal variables  
%           These variables are assumed to have rounded Gaussians kernels
% catvars: (p3x1) string vector listing categorical variables  
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
% MIout: 0/1 or false/true variable for computing pairwise empirical 
%        mutual information 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(convars); convars = string(convars); end
if isempty(tranvars); tranvars = string(tranvars); end
if isempty(compvars); compvars = string(compvars); end
if isempty(couvars); couvars = string(couvars); end
if isempty(catvars); catvars = string(catvars); end

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

    outlabs = [convars;tranvars;complabs;couvars;catvars];
    yst = [zscore(data{:,convars}),data{:,tranvars},compilrs,data{:,[couvars;catvars]}];
    

% Specify sizes and locations of the different variable types
%  Note: p1, p2 contain continuous and zero inflated variables from zerovars and compvars

    contidx = find(ismember(outlabs,convars));
    p1 = numel(contidx);
    tranidx = find(ismember(outlabs,[tranvars;complabs]));
    p2 = numel(tranidx);
    countidx = find(ismember(outlabs,couvars));
    p3 = numel(countidx);
    catidx = find(ismember(outlabs,catvars));
    p4 = numel(catidx);
    
    [n,p] = size(yst);
    p0 = p-p4;

%     outtab = table(yst);
%     outtab = splitvars(outtab);
%     outtab.Properties.VariableNames = outlabs;

% The following defines the effective sample size to be output    
    eff_samp1 = (nrun-burn)/thin;

%-- Preallocate placeholders for posterior sampling
    jointalloc = zeros(eff_samp1,n,chains,'uint8');
    infout = deal(zeros(eff_samp1,p*(p-1)/2,chains));
    lambdaout = zeros(eff_samp1,K0,chains);
    atomsmnsout = zeros(eff_samp1 , p0 , chains);
    marg = zeros(n,p);    
    
%-- Preallocate memory for tensor arm items by variable
    [Psi,nj] = deal(cell(p4,1));
%     prbs = zeros(n,K0,p);
    for jj=1:p4
        j = catidx(jj);
        nj{jj} = zeros(max(yst(:,j)),K0,'uint16');
        Psi{jj} = zeros(max(yst(:,j)),K0);
    end        

%-- Specifying the initial number of occupied classes each mixture model
% component. Different intialization at each chain. 
    d0 = floor(linspace(K0lbd,K0,chains));
    d0 = d0(randsample(chains,chains));    
    
%-- Specifying base measure hyperparameters
    [asig0,bsig0,tau] = deal(ones(p0,1));
    mu0 = zeros(p0,1);
    sig20 = 1000*ones(p0,1);
    
    asig0(contidx) = 2;
    bsig0(contidx) = 4;
    
    asig0(tranidx) = 2.5;
    bsig0(tranidx) = var(yst(:,tranidx))';
    mu0(tranidx) = mean(yst(:,tranidx))';      
    
    asig0(countidx) = 2;
    bsig0(countidx) = 1;
    tau(countidx) = var(yst(:,countidx))';
    mu0(countidx) = mean(yst(:,countidx))';
    
%-- Allocate placeholders for the base measure atom mean and variances
    muInt = zeros(p0 , chains);

%-- Initialize mu
    for jj=1:p0
        muInt(jj,:) = linspace(quantile(yst(:,jj),0.025) , quantile(yst(:,jj),0.975) , chains);
        muInt(jj,:) = muInt(jj,randsample(chains,chains));
    end

%-- Initializing bounds for count variable used in the rounded gaussian
%kernels. Note these are only valid for count variables.
    lbd = yst(:,countidx)-1;lbd(lbd == min(lbd)) = -Inf;
    ubl = yst(:,countidx);ubl(ubl == max(ubl)) = Inf;

    npdf = 100;
    pdfidx = sort(randsample(n , npdf));
    ipdfout = zeros(eff_samp1 , npdf , chains);    
    
%-- Starting Gibbs sampler
tic;
for c = 1:chains
    fprintf('BayesMPK: Chain %d\n',c);

    % Initilize joint allocation variable
    
    z = zeros(n,1);
    nidx = randsample(n,n);
    nsz = gamrnd(ones(d0(c),1),1);
    nsz = mnrnd(n,nsz./sum(nsz));
    nsz = [0;cumsum(nsz)'];
    for ll = 1:d0(c)
        z(nidx((nsz(ll)+1):nsz(ll+1))) = ll;
    end
    
    yst(:,countidx) = data{:,cellstr(couvars)} - rand([n , p3]);
    
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
        
        %-- Update parameters for means and variances for marginal
        %distributions        
        taul = tau + nz';
        pl = tau./taul;
        ymnv = accumarray([reshape(repmat(1:p0,n,1),[],1),repmat(z,p0,1)],...
            reshape(yst(:,1:p0),[],1),[],@mean);
        Sl2 = (nz-1)'.*accumarray([reshape(repmat(1:p0,n,1),[],1),repmat(z,p0,1)],...
            reshape(yst(:,1:p0),[],1),[],@var);
        mnv = pl.*mu + (1-pl).*ymnv;        
        sig2v = bsig0 + Sl2/2 + taul.*pl.*(1-pl).*(ymnv-mu).^2/2;
        asigl = asig0 + nz'/2;
        sig2v = 1./gamrnd(asigl , 1./sig2v);
        mnv = normrnd(mnv , sqrt(sig2v./taul)); 
        
        % Update latent count variables (count variables) 
        for jj = 1:p3       
            j = countidx(jj);
            pytmp = ([lbd(:,jj) ubl(:,jj)] - mnv(j,z)')./sqrt(sig2v(j,z))';
            yst(:,j) = mnv(j,z)' + sqrt(sig2v(j,z))'.*trandn(pytmp(:,1) , pytmp(:,2));            
        end

        cprobs = ones(n,K0);        
        prbs = zeros(n,K0,p);
        %-- Update categorical variable components (arms)
        for jj = 1:p4
            j = catidx(jj);
            nj{jj} = accumarray([yst(:,j),z] , 1);
            Psi{jj} = gamrnd(nj{jj} + 1 ,1);
            Psi{jj} = Psi{jj}./sum(Psi{jj});
            prbs(:,:,j) = Psi{jj}(yst(:,j),:);
            cprobs = cprobs.*prbs(:,:,j);
        end
        
        %-- Update allocations
        for j = 1:p0
            if (ismember(j,countidx))
                jj = j - p1 - p2;
%                 prbs(:,:,j) = normcdf((ubl(:,jj) - mnv(j,:))./sqrt(sig2v(j,:))) - ...
%                     normcdf((lbd(:,jj) - mnv(j,:))./sqrt(sig2v(j,:)));
                prbs(:,:,j) = pnorm2_mex([lbd(:,jj),ubl(:,jj)] , mnv(j,:) , sqrt(sig2v(j,:)));
            else
                prbs(:,:,j) = normpdf(yst(:,j) , mnv(j,:) , sqrt(sig2v(j,:))); 
            end            
            cprobs = cprobs.*prbs(:,:,j);
        end
        zupdateprob = lambda'.*cprobs;
%         upidx = sum(zupdateprob == 0,2) ~= K0; 
        zupdateprob = zupdateprob./sum(zupdateprob,2);
        upidx = all(~isnan(zupdateprob),2);
        zupdateprob1 = [zeros(n,1) cumsum(zupdateprob,2)];
        rr = rand(n,1);
        [ztmp,~] = find((rr>=zupdateprob1(:,1:(end-1)) & rr < zupdateprob1(:,2:end))');      
        z(upidx) = ztmp;
        ipdf = cprobs(pdfidx, :)*lambda;        
        
        %-- Update base measure mean hyper parameters
        sigtilde = 1./(tau.*sum(1./sig2v,2) + 1./sig20);
        mutilde = sigtilde.*(tau.*sum(mnv./sig2v,2) + mu0./sig20);
        mu = normrnd(mutilde , sqrt(sigtilde));
        
        if (numel(unique(z))<2); break; end

        % -- positional dependence -- %
        if (mod(b - burn,thin) == 0 && b > burn)
           if logical(MIout)
               ct_loop = 0;
               for j = 1:p
                   marg(:,j) = prbs(:,:,j)*lambda;
               end

               for j1 = 1:(p-1)
                   for j2 = (j1+1):p                   
                       ct_loop = ct_loop + 1;
                       jointprobs = (prbs(:,:,j1).*prbs(:,:,j2))*lambda;
                       infout((b-burn)/thin,ct_loop,c) = mean(log(jointprobs./(marg(:,j1).*marg(:,j2))));            
                   end
               end
           end
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

