function Sxint = gsua_intrp(sol,xint,idx,extra)
if nargin <4
    extra=1;
end
if extra==1
    if ~isa(sol,'struct')  
      % Try  DEVAL(XINT,SOL)
      temp = struct();
      sol  = xint;
      xint = temp;
    else
        try
          t = sol.x;    
          y = sol.y; 
          if isfield(sol,'solver')
              solver=sol.solver;
          else
              solver='';
          end
          reps=length(idx);
          if any(strcmp(solver,{'dde23',''}))
              a=zeros(1,size(y,1),size(y,2));
              a(1,:,:)=y;
              y=a;
          end

        catch
          error(message('MATLAB:deval:SolNotFromDiffEqSolver', inputname( 1 )));
        end

    end


    sz=size(y);
    if nargin < 3
      idx = 1:sz(2);  % return all solution components
    else 
      if any(idx < 0) || any(idx > size(y,2))
        error(message('MATLAB:deval:IDXInvalidSolComp', inputname( 3 )));
      end  
    end  

    if reps==1 
        if sz(1)>1
            Sxint=interp1(t,squeeze(y(:,idx,:))',xint)';
        else
            Sxint=interp1(t,squeeze(y(:,idx,:))',xint);
        end
    else
        Sxint=zeros(sz(1),length(xint),reps);
        for i=1:reps
            Sxint(:,:,i)=interp1(t,squeeze(y(:,idx(i),:))',xint)';
        end
    end
else
    t = sol.x;
    y = sol.y;
    reps=length(idx);
    Sxint=zeros(reps,length(xint));
    for i=1:reps
        Sxint(i,:)=interp1(t,y(idx(i),:),xint);
    end
end