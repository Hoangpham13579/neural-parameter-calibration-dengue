function cos=gsua_costfMulti(ydata,yfunction,margin,parallel,alpha)
if nargin<3
    margin=0.1;
    parallel=false;
end
if nargin<5
    alpha=1;
end
if parallel
    parforArg=Inf;
else
    parforArg=0;
end
if margin<=1
    margin=1+abs(margin);
end

[reps,len,inputs]=size(ydata);
if size(yfunction,3)>1
    yfunction=squeeze(yfunction)';
end
regulator=sum((yfunction-yfunction*margin).^2,2)/len;
cos=zeros(1,reps);

parfor (j=1:reps,parforArg) 
%for j=1:reps
    cost=zeros(1,inputs);
    try
        for i=1:inputs
            icorre=(2-corr2(ydata(j,:,i),yfunction(i,:)));
            if isnan(icorre)
                icorre=1;
            end
            cost(i)=(icorre*(sum((ydata(j,:,i)-yfunction(i,:)).^2,2)/len)/regulator(i))^alpha;
        end
        cos(j)=sum(cost)/inputs;
    catch
        cos(j)=inf;
    end
    
%     if isnan(cos(j))
%         disp('fail')
%     end
end
end