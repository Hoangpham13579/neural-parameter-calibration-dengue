function [depth,idx]=gsua_depth(x,parallel)
%Calcula profundidad por bandas generalizada de x con respecto a x
if nargin<2
    parallel=false;
end
if parallel
    parforArg=Inf;
else
    parforArg=0;
end

[n,d,inputs]=size(x);

depth=zeros(n,inputs);

for h=1:inputs
    PosiXX=zeros(n,d);
    parfor (i=1:n,parforArg)
        for j=1:d
            PosiXX(i,j)=sum(x(:,j,h)<x(i,j,h))+1;
        end
    end
    %disp(PosiXX)
    %disp(x);
    contador=n*(n+1)/2;
    parfor (i=1:n,parforArg)
        for j=1:d
            depth(i,h)=depth(i,h)+(PosiXX(i,j)-1)*(n-PosiXX(i,j))+n;
        end
    end

    depth(:,h)=depth(:,h)/contador;
end
depth=mean(depth,2);
[~,idx]=sort(depth,'descend');
end
        