function T = gsua_ia(T,T_est,correction,outlier,show)

if nargin<5
    show=false;
end
if nargin<4
    outlier=false;
end
if nargin <3
    correction=false;
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
D1 = floor(sqrt(size(T,1))); % Number of rows of subplot
D2 = D1+ceil((size(T,1)-D1^2)/D1);
figure('Name','CDF Range')
for i=1:size(T,1)
subplot(D1,D2,i)
ecdf(T.Est(i,:),'Bounds','on')
title(T.Properties.RowNames{i})
end

Normalized=zeros(size(T,1),size(T.Est,2));
for i=1:size(T,1)
    Normalized(i,:)=(T.Est(i,:)-T.Range(i,1))/(T.Range(i,2)-T.Range(i,1));
end
nnominal=(T.Nominal-T.Range(:,1))./(T.Range(:,2)-T.Range(:,1));
figure('Name','Normalized Practical identifiability')
clf
boxplot(Normalized','Labels',T.Properties.RowNames,'LabelOrientation','inline')
h = findobj(gca, 'type', 'text');
set(h, 'Interpreter', 'tex','FontWeight','bold');

B = filloutliers(Normalized','center','median');
indicator=max(B,[],1)-min(B,[],1);
[~,index]=sort(indicator,'descend');
nn=B(:,index);
names=T.Properties.RowNames(index);
figure('Name','Sorted parameter range')
clf
boxplot(nn,'Labels',names,'LabelOrientation','inline')
h = findobj(gca, 'type', 'text');
set(h, 'Interpreter', 'tex','FontWeight','bold');
hold on
nnominal=nnominal(index);
plot(nnominal,'.','MarkerSize',20,'Color','black')

figure('Name','Correlation')
RHO = corr(T.Est');
imagesc(RHO)
xticks(1:size(T,1))
xticklabels(T.Properties.RowNames)
%xtickangle(60)
yticks(1:size(T,1))
yticklabels(T.Properties.RowNames)
caxis([-1 1])
colormap jet
colorbar
% figure('Name','NPI without outliers')
% clf
% boxplot(B,'Labels',T.Properties.RowNames,'LabelOrientation','inline')
% 

x=T.Est;
med=median(x,2);
N=size(x,2);
desv=std(x,[],2);

lb = med-1.96*sqrt(pi/2)*desv/sqrt(N);
up = med+1.96*sqrt(pi/2)*desv/sqrt(N);

if (any(lb(lb<T.Range(:,1))))||(any(up(up>T.Range(:,2))))
    %action=input('Replace out ranges by estimation ranges? (true,false): ');
    action=correction;
    if action
        lb(lb<T.Range(:,1))= T.Range(lb<T.Range(:,1),1);
        up(up>T.Range(:,2))= T.Range(up>T.Range(:,2),2);
    end
end


boxin=(up-lb)./(T.Range(:,2)-T.Range(:,1));
len=length(boxin);
corrin=sum(abs(RHO))/len;
extrin=sum(abs(RHO)>0.5)/len;
ind=(2*boxin'+corrin+extrin)/4;
[~,idx]=sort(ind,'descend');

if show

    figure('Name','Identifiability Index')
    scatter3(boxin(idx(1)),corrin(idx(1)),extrin(idx(1)),60,ind(idx(1)),'filled','DisplayName',T.Properties.RowNames{idx(1)})
    hold on
    for i=idx(2:end)
        scatter3(boxin(i),corrin(i),extrin(i),60,ind(i),'filled','DisplayName',T.Properties.RowNames{i})
    end
    colormap jet
    % b=colorbar('Location','eastoutside');
    % b.Label.String = 'Identifiability Index';
    legend('NumColumns',2','Orientation','vertical')
    xlabel('Interval Index')
    ylabel('Correlation Index')
    zlabel('Strong correlation Index')


    figure('Name','Identifiability graph')
    G=graph((abs(RHO)>0.5)-eye(len),T.Properties.RowNames);
    G.Nodes.Weights=ind';
    G.Nodes.NodeColors = ind';
    p=plot(G);
    p.NodeCData = G.Nodes.NodeColors;
    p.NodeFontSize=12;
    p.MarkerSize=7;
    colormap jet
    a = colorbar;
    a.Label.String = 'Identifiability Index';
    title('Strong correlations')
end

T.Range=[lb,up];
T.index=ind';
T.Nominal=med


% indicator=[max(B,[],1)-min(B,[],1)];
% [~,index]=sort(indicator);
% nn=B(:,index);
% names=T.Properties.RowNames(index);
% val=median(nn,1);
% nn=[nn-val];
% nnominal=nnominal(index)'-val;
% figure('Name','Sorted parameter range')
% boxplot(nn,'Labels',names,'LabelOrientation','inline')
% hold on
% plot(nnominal,'Marker','*','MarkerSize',2)
end