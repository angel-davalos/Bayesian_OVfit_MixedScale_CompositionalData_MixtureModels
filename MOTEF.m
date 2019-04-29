function [jointalloc,outlabs,margalloc,atomsmnsout,time,infout,ipdfout,lambdaout] = MOTEF(data , convars , zerovars , compvars , ...
    couvars , catvars , chains , burn , nrun , thin , K0 , K0lbd , ...
    K , Klbd , alpha0 , alpha , MIout)
%Summary: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Variable Input descriptions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% convars: (p1x1) string vector listing continuous variables 
%           These variables are assumed to marginally be overfitted mixture of normals
% zerovars: (p2x1) string vector listing continuous variables 
%              These variables are assumed to marginally be zero-inflated overfitted mixture of normals
%              zero-inflated positive support variables (log transformed)
% compvars: (kx1) cell of string vectors each listing a unique composition
%          The referent composition element should be the first element
%          listed in each composition
% couvars: (p3x1) string vector listing count/ordinal variables  
%           These variables are assumed to marginally be overfitted mixture
%           of rounded Gaussians
% catvars: (p3x1) string vector listing categorical variables  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Algorithm Input specifics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% chains: The number of chains to run (recommended 5)
% burn: The number intial sampling iterations to be discarded
% nrun: The total number of iterations to be run
% thin: The thinning interval (record the ith iteration)
% K0: The maximum size of the tensor joint mixture component
% K0lbd: The lower bound of the size of the tensor joint mixture component
% K: The maximum size of each marginal mixture component
% Klbd: The lower bound of the size of each marginal mixture component
% alpha0: The symmetric Dirichlet mixture prior on the tensor joint mixture
% alpha: The symmetric Dirichlet mixture prior for each marginal model


if isempty(convars); convars = string(convars); end
if isempty(zerovars); zerovars = string(zerovars); end
if isempty(compvars); compvars = string(compvars); end
if isempty(couvars); couvars = string(couvars); end
if isempty(catvars); catvars = string(catvars); end

% Creating the analysis variables

    ncomps = numel(compvars);
    njcomp = zeros(ncomps , 1);
    compalrs = cell(1,ncomps); 
    complabs = cell(ncomps,1);
    for cc = 1:ncomps
        calr = data{:,cellstr(compvars{cc})};
        calr = calr./sum(calr,2);
        compalrs{cc} = log(calr(:,2:end)./calr(:,1));
        [~,njcomp(cc)] = size(compalrs{cc});
%         p2 = p2 + njcomp(cc);
        complabs{cc} = string(compose('%s_%s' , compvars{cc}(2:end) , compvars{cc}(1)));
    end    
    compalrs = cell2mat(compalrs);
    complabs = vertcat(complabs{:});

    outlabs = [convars;zerovars;complabs;couvars;catvars];
    yst = [zscore(data{:,convars}),log(data{:,zerovars}),compalrs,data{:,[couvars;catvars]}];
    

% Specify sizes and locations of the different variable types
%  Note: p1, p2 contain continuous and zero inflated variables from zerovars and compvars

    contidx = find(ismember(outlabs,convars));
    p1 = numel(contidx);
    zncidx = find(ismember(outlabs,[zerovars;complabs]));
    p2 = numel(zncidx);
    countidx = find(ismember(outlabs,couvars));
    p3 = numel(countidx);
    catidx = find(ismember(outlabs,catvars));
    p4 = numel(catidx);
    zeroidx = find(any(isinf(yst)))';
    
    [n,p] = size(yst);
    p0 = p-p4;

%     outtab = table(yst);
%     outtab = splitvars(outtab);
%     outtab.Properties.VariableNames = outlabs;

% The following defines the effective sample size to be output    
    eff_samp1 = (nrun-burn)/thin;
    
% Specifying the size of each mixture model component
    K = [K*ones(p0,1);max(yst(:,catidx))']; % size for each arm of the tensor (each marginal mixture)
    Klbd = Klbd*ones(p0,1);

%-- Preallocate placeholders for posterior sampling
    margalloc = zeros(eff_samp1,n,p0,chains,'uint8');
    jointalloc = zeros(eff_samp1,n,chains,'uint8');
    infout = deal(zeros(eff_samp1,p*(p-1)/2,chains));
    marg = zeros(n,p);
    
%-- Preallocate memory for tensor arm items by variable
    [Psi,nj] = deal(cell(p,1));
    [Omega,prbs] = deal(cell(p0,1));
    for j=1:p0
        nj{j} = zeros(K(j),K0,'uint16');
        Psi{j} = zeros(K(j),K0);
    end
    lambdaout = zeros(eff_samp1,K0,chains);
    atomsmnsout = zeros(eff_samp1 , p0 , chains);

%-- Specifying the initial number of occupied classes each mixture model
% component. Different intialization at each chain. 
    d0 = floor(linspace(K0lbd,K0,chains));
    d0 = d0(randsample(chains,chains)); 
    d = ones(p0,chains);
    for j=1:p0
        tmp = floor(linspace(Klbd(j),K(j),chains));
        d(j,:) = tmp(randsample(chains,chains,false));
    end      
    
%-- Specifying base measure hyperparameters
    [asig0,bsig0,tau] = deal(ones(p0,1));
    mu0 = zeros(p0,1);
    sig20 = 1000*ones(p0,1);
    
    asig0(contidx) = 2;
    bsig0(contidx) = 4;
    
    asig0(zncidx) = 2.5;
    for j = zncidx'
        cidx = ~isinf(yst(:,j));
        bsig0(j) = var(yst(cidx,j));
        mu0(j) = mean(yst(cidx,j));
    end            
    
    asig0(countidx) = 2;
    bsig0(countidx) = 1;
    tau(countidx) = var(yst(:,countidx))';
    mu0(countidx) = mean(yst(:,countidx))';
    
%-- Allocate placeholders for the base measure atom mean and variances
    [muj,sig2j] = deal(cell(p0,1));

    muInt = zeros(p0 , chains);

%-- Initialize mu
    for jj=1:p0
        cidx = ~isinf(yst(:,jj));
        muInt(jj,:) = linspace(quantile(yst(cidx,jj),0.025) , quantile(yst(cidx,jj),0.975) , chains);
        muInt(jj,:) = muInt(jj,randsample(chains,chains));
    end

%-- Initializing bounds for count variable used in the rounded gaussian
%kernels. Note these are only valid for count variables.
    lbd = yst(:,countidx)-1;lbd(lbd == min(lbd)) = -Inf;
    ubl = yst(:,countidx);ubl(ubl == max(ubl)) = Inf;

    npdf = 100;
    pdfidx = sort(randsample(n , npdf));
    ipdfout = zeros(eff_samp1 , npdf , chains);    
    
% Initialize marginal allocations for each variable to be categorized
    X = zeros(n,p,'uint8');
    X(:,catidx) = yst(:,catidx);      
    
%-- Starting Gibbs sampler
tic;
for c = 1:chains
    fprintf('MOTEF: Chain %d\n',c);

    % Initilize joint allocation variable
    
    z = zeros(n,1);
    nidx = randsample(n,n);
    nsz = gamrnd(ones(d0(c),1),1);
    nsz = mnrnd(n,nsz./sum(nsz));
    nsz = [0;cumsum(nsz)'];
    for ll = 1:d0(c)
        z(nidx((nsz(ll)+1):nsz(ll+1))) = ll;
    end
    
    for j = 1:p0
        for tt = unique(z)'
            nidx = find((z==tt).*(~isinf(yst(:,j))));
            nn = length(nidx);
            nidx = nidx(randsample(nn,nn));
            nsz = gamrnd(ones(d(j,c)-1,1),1);
            nsz = mnrnd(nn,nsz./sum(nsz));
            nsz = [0;cumsum(nsz)'];        
            for ll = 1:(d(j,c)-1)
                X(nidx((nsz(ll)+1):nsz(ll+1)),j) = ll;
            end
        end
        if(ismember(j,zeroidx))
            X(:,j) = X(:,j) + 1;
        end
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
        
        rr = rand(n,p0);                
        cprobs = ones(n,K0);

        for j=1:p0
            [X(:,j),~] = find((X(:,j) == unique(X(:,j))')');
            K(j) = max(X(:,j));
            nj{j} = accumarray([X(:,j),z] , 1);
            Psi{j} = gamrnd(nj{j} + 1 ,1);
            Psi{j} = Psi{j}./sum(Psi{j});
            nkj = sum(nj{j},2);
            Omega{j} = gamrnd(nkj+alpha,1);            
            Omega{j} = Omega{j}./sum(Omega{j});
            
            kap0 = tau(j);  
            ymnv = accumarray(X(:,j),yst(:,j))./nkj;
            Sl2 = (nkj - 1).*accumarray(X(:,j),yst(:,j) , [] , @var);
            if (ismember(j,zeroidx))
                nkj = nkj(2:end);
                ymnv = ymnv(2:end);
                Sl2 = Sl2(2:end);
            end
        % Update means and variances
            asigt = asig0(j) + nkj/2;
            pl = kap0./(kap0 + nkj);
            bsigt = bsig0(j) + Sl2/2 + (kap0 + nkj).*pl.*(1-pl).*(ymnv-mu(j)).^2/2;
            sig2j{j} = 1./gamrnd(asigt , 1./bsigt);
            mnt = pl*mu(j) + (1-pl).*ymnv;
            muj{j} = normrnd(mnt , sqrt(sig2j{j}./(kap0 + nkj)));  
            
        %-- Update latent continuous variable for rounded gaussian kernels
            if(ismember(j,countidx))
                jj = j-p1-p2;
                pytmp = ([lbd(:,jj) ubl(:,jj)] - muj{j}(X(:,j)))./sqrt(sig2j{j}(X(:,j)));
                yst(:,j) = muj{j}(X(:,j)) + sqrt(sig2j{j}(X(:,j))).*trandn(pytmp(:,1) , pytmp(:,2));
                                            
        %-- Update marginal allocation variables  
%                 prbs{j} = normcdf((ubl(:,jj) - muj{j}')./sqrt(sig2j{j}')) - ...
%                     normcdf((lbd(:,jj) - muj{j}')./sqrt(sig2j{j}'));                
                prbs{j} = pnorm2_mex([lbd(:,jj),ubl(:,jj)] , muj{j}' , sqrt(sig2j{j}'));                
            else 
        %-- Update allocations for zero-inflated/gaussian kernels
                prbs{j} = normpdf(yst(:,j) , muj{j}' , sqrt(sig2j{j}'));                
                if(ismember(j,zeroidx))
                    tprbs = prbs{j};
                    prbs{j} = [isinf(yst(:,j)) , tprbs];
                end
            end
            zupdateprob = Omega{j}'.*prbs{j};
%             upidx = sum(zupdateprob == 0,2) ~= K(j);
            zupdateprob = zupdateprob./sum(zupdateprob,2);
            upidx = all(~isnan(zupdateprob),2);
            zupdateprob1 = [zeros(n,1) cumsum(zupdateprob,2)];
            [Xtmp,~] = find((rr(:,j)>=zupdateprob1(:,1:(end-1)) & rr(:,j) < zupdateprob1(:,2:end))');    
            X(upidx,j) = Xtmp;
        
        %-- Update base measure mean hyper parameters
            sigtilde = 1/(tau(j)*sum(1./sig2j{j}) + 1/sig20(j));
            mutilde = sigtilde*(tau(j)*sum(muj{j}./sig2j{j}) + mu0(j)/sig20(j));
            mu(j) = normrnd(mutilde , sqrt(sigtilde));
            
            cprobs = cprobs.*Psi{j}(X(:,j),:);
        end
        

        %-- Update joint allocation variables
        for j = catidx'
            nj{j} = accumarray([X(:,j),z] , 1);
            Psi{j} = gamrnd(nj{j} + 1 ,1);
            Psi{j} = Psi{j}./sum(Psi{j});
            cprobs = cprobs.*Psi{j}(X(:,j),:);
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
        
        if(numel(unique(z))<2);break;end

        % -- positional dependence -- %
        if (mod(b-burn,thin) == 0 && b > burn)
           if logical(MIout)
               ct_loop = 0;
               for j = 1:p0
                   marg(:,j) = prbs{j}*Psi{j}*lambda;
               end
               for j = catidx'
                   marg(:,j) = Psi{j}(X(:,j),:)*lambda;
               end

               for j1 = 1:(p-1)
                   for j2 = (j1+1):p                   
                       ct_loop = ct_loop + 1;
                       Probtnj1j2 = Psi{j1}*diag(lambda)*Psi{j2}';
                       if ismember(j1,catidx)
                           if ismember(j2,catidx)
                               Probtnj1j2i = Probtnj1j2(sub2ind([K(j1),K(j2)],X(:,j1),X(:,j2)));
                           else                           
                               Probtnj1j2i = sum(prbs{j2}.*Probtnj1j2(X(:,j1),:),2);
                           end
                       else                       
                           if ismember(j2,catidx)
                               Probtnj1j2i = sum(prbs{j1}.*Probtnj1j2(:,X(:,j2)')',2);
                           else
                               Probtnj1j2i = zeros(n,1);
                               for c1 = 1:K(j1)
                                   for c2 = 1:K(j2)
                                       Probtnj1j2i = Probtnj1j2i + Probtnj1j2(c1,c2)*prbs{j1}(:,c1).*prbs{j2}(:,c2);
                                   end
                               end
%                                [ri,ci] = ind2sub([K(j1),K(j2)],1:(K(j1)*K(j2)));
%                                Probtnj1j2i = (prbs{j1}(:,ri).*prbs{j2}(:,ci))*Probtnj1j2(:);
                           end
                       end
                       infout((b-burn)/thin,ct_loop,c) = mean(log(Probtnj1j2i./(marg(:,j1).*marg(:,j2))));            
                   end
               end
           end
           for j = 1:p0
               margalloc((b-burn)/thin,:,j,c) = X(:,j)';
           end
           lambdaout((b-burn)/thin,1:K0,c) = lambda';
        %    armsout((b-burn)/thin,:,c) = reshape(cell2mat(Psi),1,[],K0);
           jointalloc((b-burn)/thin,:,c) = z';
           atomsmnsout((b-burn)/thin,:,c) = mu';           
           ipdfout((b-burn)/thin,:,c) = ipdf';           
        end
        if mod(b,nrun/10) == 0, fprintf('%d%% Complete.\n',b/nrun*100); end
    end   

end
time = toc;

end

% The following functions were written by another programmer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: trandn.m code
% Author: Zdravko Botev
% Date: 2016
% Code version: 1.0
% https://www.mathworks.com/matlabcentral/fileexchange/53180-truncated-normal-generator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


