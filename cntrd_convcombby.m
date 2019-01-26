function ctrds_convcomb = cntrd_convcombby(y,optclu)
%cntrd_convcomb Produces the convex combination centroid of a compositional
%data set
%   y: n x D matrix of compositional data 
    conf = grp2idx(categorical(join(string((y == 0)*1),'')));
% Overall
    [rr,cc] = size(y);
    pages = repmat(conf,cc,1);
    cols = reshape(repmat(1:cc,rr,1),[],1);
    rows = repmat(optclu,cc,1);
    
% Grouped convex combination    
    props = accumarray([rows,pages] , 1);
    props = props./sum(props,2);
    ctrds_convcomb = accumarray([rows,cols,pages] , y(:) , [] , @geomean);
    ctrds_convcomb = ctrds_convcomb./sum(ctrds_convcomb,2);
    ctrds_convcomb = nansum(reshape(props,[],1,size(props,2)).*ctrds_convcomb,3);

end

