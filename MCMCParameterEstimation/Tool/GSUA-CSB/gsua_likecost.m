function cost=gsua_likecost(ydata,ynom,margin)

margin=1+abs(margin);
ymargin=ynom*margin;
desv = abs((ymargin).^2);
[reps,~,~]=size(ydata);
cost = zeros(1,reps);
costmargin = sum(log(2*pi*desv) +(ymargin-ynom).^2./desv,'all','omitnan')/2;

for j=1:reps
    cost(j)=sum(log(2*pi*desv) +(ydata(j,:,1)-ynom).^2./desv,'all','omitnan')/2;
end
cost=cost-costmargin;
end