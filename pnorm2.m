function probout = pnorm2(X,mu,sig)
%pnorm bounded evaluation of normcdf
%   Detailed explanation goes here

% persistent n
% if isempty(n)    
    n = size(X,1);
% end



p = numel(mu);

probout = zeros(n,p);

    for j = 1:p
        probout(:,j) = (erfc(-((X(:,2)-mu(j))./sig(j))/sqrt(2)) - ...
            erfc(-((X(:,1)-mu(j))./sig(j))/sqrt(2)))/2;
    end
end

