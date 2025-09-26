function [Jsup,Jinf] = gsua_MCF(T,M,Y,y_exp,tIn,tOut)
 
try
     fixed=T.Properties.CustomProperties.Fixed;
 catch
     TP=load('ATable.mat');
     TP=TP.Table2;
     fixed=TP.Fixed;
end

if isempty(fixed)
 fixed=zeros(1,length(M(1,:)));
end

M=M(:,~fixed);
names=T.Properties.RowNames(~fixed);

[N,Np]=size(M);


% 
D1 = floor(sqrt(Np)); % Number of rows of subplot
D2 = D1+ceil((Np-D1^2)/D1); % Number of columns of subplot
Jinf=cell(Np,1);
Jsup=cell(Np,1);

dimension=size(Y,1);
try
    if nargin<5

        Jexp=sum(y_exp);
        if dimension>1
            J=sum(Y,2);
        else
            J=Y;
        end
        t_in=0;

        for i=1:Np
            Jinf{i}=M(J<Jexp,i);
            Jsup{i}=M(J>Jexp,i);
            if isempty(Jinf{i})
              Jinf{i}=min(M(:,i));  
            end
            if isempty(Jsup{i})
              Jsup{i}=max(M(:,i));  
            end
        end
    else
        t_in=find(tOut==tIn);
        Jexp=(y_exp(t_in));
        J=(Y(:,t_in));
        for i=1:Np
            Jinf{i}=M(J<Jexp,i);
            Jsup{i}=M(J>Jexp,i);
        end
        if isempty(Jinf{i})
          Jinf{i}=min(M(:,i));  
        end
        if isempty(Jsup{i})
          Jsup{i}=max(M(:,i));  
        end
    end
        
for k = 1:Np
             subplot(D1,D2,k);
             cdfplot(M(:,k))
             hold on
             cdfplot(Jinf{k})
             cdfplot(Jsup{k})
             xlabel('Value')
             ylabel('CDF')
             title(names{k})
             hold off
end
         legend({'Prior','Low Values','High Values'})
         if t_in==0
             
            h = title(axes,{['Montecarlo Filtering for escalar Y with N: ' num2str(N)];' '},'Color','r');
         else
            h = title(axes,{['Montecarlo Filtering with N: ' num2str(N) ' in t: ' num2str(tIn)];' '},'Color','r');
         end
         set(gca,'visible','off')
         set(h,'visible','on')
catch ME
    disp(ME.message)
    return
end

end