function T = gsua_dia(T,T_est,outlier)

if nargin <3
    outlier=false;
end
if outlier
    disp('Removing outliers...')
    [~,~,RD,chi_crt]=DetectMultVarOutliers(T_est');
    id_in=RD<chi_crt(4);
    T_est=T_est(:,id_in);
    disp(num2str(sum(id_in))+" outliers were removed")
end
T.Est=T_est;
T.Nominal=T.Est(:,1);


Normalized=zeros(size(T,1),size(T.Est,2));
for i=1:size(T,1)
    Normalized(i,:)=(T.Est(i,:)-T.Range(i,1))/(T.Range(i,2)-T.Range(i,1));
end

RHO = corr(T.Est');

x=T.Est;
med=median(x,2);
N=size(x,2);
desv=std(x,[],2);

lb = med-1.96*sqrt(pi/2)*desv/sqrt(N);
up = med+1.96*sqrt(pi/2)*desv/sqrt(N);

if (any(lb(lb<T.Range(:,1))))||(any(up(up>T.Range(:,2))))
    lb(lb<T.Range(:,1))= T.Range(lb<T.Range(:,1),1);
    up(up>T.Range(:,2))= T.Range(up>T.Range(:,2),2);   
end


boxin=(up-lb)./(T.Range(:,2)-T.Range(:,1));
len=length(boxin);
corrin=sum(abs(RHO))/len;
extrin=sum(abs(RHO)>0.5)/len;
ind=(2*boxin'+corrin+extrin)/4;

T.Range=[lb,up];
T.index=ind';
T.Nominal=med;


end