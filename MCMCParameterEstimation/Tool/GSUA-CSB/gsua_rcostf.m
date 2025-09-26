function cos=gsua_rcostf(ydata,yfunction,margin,alpha)
if nargin<3
    margin=1.1;
end
if nargin<4
    alpha=2;
end

margin=abs(margin);

if margin<1
    margin=margin+1;
end

[inputs,lon] = size(ydata);
regulator=sum((ydata-ydata*margin).^2,2)/lon;
cost=(sum((ydata-yfunction).^2,2)/lon)./regulator;


for i=1:inputs
    factor = 2-corr2(ydata(i,:),yfunction(i,:));
    if isnan(factor)
        factor = 1;
    end
    cost(i)=(factor*cost(i))^alpha;
end
cos=sum(cost)/inputs;
end