function [optclu , optitr] = minBinderLoop2(pclus)
%Summary: minBinder is a function which finds the clustering point estimate
%         from a posterior sample of clusterings based off of the
%         minimizing Binder's posterior expected loss. (Dahl 2006, Wade
%         Gharamani 2018)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Variable Input descriptions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plcus: (B x n) matrix of posterior clusterings where each row is a 
%                clustering -or- 
%        (B x n x c) array of posterior clusterings where each row is a 
%                    clustering across c chains (page, third dimension)
% psm: (n x n) upper triangular matrix of posterior simmilarity matrix
%              each element i,j (i<j) is the posterior probability i , j 
%              are clustered together. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Variable Output descriptions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% optclu: (n x 1) optimal clustering
% optitr: optimal iteration 
% psm: (n x n) upper triangular matrix of posterior simmilarity matrix
%              each element i,j (i<j) is the posterior probability i , j 
%              are clustered together. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    
    n = size(pclus,2);
    if numel(size(pclus)) == 3
        pclus = reshape(permute(pclus,[1,3,2]),[],n);
    end
    nitr = size(pclus,1);
    
%     if isempty(psm)
%         psm = zeros(n,n);
%         fprintf('Computing Post Sim Matrix.\n'); 
%         for bb = 1:nitr
%             clvl = unique(pclus(bb,:));
%             for kk = 1:numel(clvl)
%                 k = clvl(kk);
%                 cidx = nchoosek(find(pclus(bb,:) == k),2);
%                 if numel(cidx)>1
%                     linidx = sub2ind([n,n] , cidx(:,1), cidx(:,2));
%                     psm(linidx) = psm(linidx) + 1;
%                 end
%             end
%             if mod(bb,nitr/10) == 0, fprintf('%.0f%% Complete.\n',bb/nitr*100); end
%         end
%         psm = psm / nitr;
%     end
    
    fprintf('Binder Loss.\n'); 
    bindr = zeros(nitr,1);
    count = 0;
    cmax = n*(n-1)/2;
    for ii = 1:(n-1)
        for jj = (ii+1):n
            count = count + 1;
            tmp = pclus(:,ii) == pclus(:,jj);
            bindr = bindr + abs(tmp - mean(tmp));
            if(~mod(count , floor(cmax*0.10)))
                fprintf('Opt. Clu. Sel. Prog.: %.2f%%\n',count/cmax*100);
            end
        end
    end
    
    [~ , optitr] = min(bindr);
    optclu = pclus(optitr,:)';
end

