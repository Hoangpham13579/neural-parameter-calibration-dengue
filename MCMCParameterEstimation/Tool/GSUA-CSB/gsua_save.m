function gsua_save(T,filename)
    if nargin<2
        filename='portable.mat';
    end
    T=rmprop(T,'Solver');
    save(filename,'T');
end