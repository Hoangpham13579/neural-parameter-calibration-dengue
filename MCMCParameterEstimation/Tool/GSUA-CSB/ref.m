function optim=ref(est,table,xdata,ydata,exe)
ex=true(1,size(table,1));
ex(exe)=false;
solver='fmincon';
opt=optimoptions('fmincon','UseParallel',false,'Display','iter','MaxFunctionEvaluations',5000);
flag=true;
while flag
    range=table.Range;
    if any(any([range(ex,2)<est(ex)*1.1,range(ex,1)>est(ex)*0.9]))
        A=find(range(:,2)<est*1.1);
        B=find(range(:,1)>est*0.9);
        A=setdiff(A,exe);
        B=setdiff(B,exe);
        table.Range(A,2)=est(A)*1.5;
        table.Range(B,1)=est(B)*0.5;
    else
        break
    end

    [table,~]=gsua_pe(table,xdata,ydata,'solver',solver,'N',1,'opt',opt,'ipoint',est');
    est=table.Estfmincon(:,1);
end   
optim=table;
end