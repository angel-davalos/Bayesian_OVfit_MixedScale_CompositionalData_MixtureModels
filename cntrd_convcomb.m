function ovctrd = cntrd_convcomb(y)
%cntrd_convcomb Produces the convex combination centroid of a compositional
%data set
%   y: n x D matrix of compositional data 
    conf = grp2idx(categorical(join(string((y == 0)*1),'')));
% Overall
    [rr,cc] = size(y);
    rows = repmat(conf,cc,1);
    cols = reshape(repmat(1:cc,rr,1),[],1);
    props = accumarray(conf , 1)/rr;
    ovctrd = accumarray([rows,cols] , y(:) , [] , @geomean);
    ovctrd = props'*(ovctrd./sum(ovctrd,2));
end

