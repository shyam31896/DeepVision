list=["ikeakaustby.obj"]
basemodel=read_wobj('model1 (2).Obj')
base_vertices=getfield(basemodel,'vertices')
base_vertices=reshape(base_vertices,3,300)
for i=1:length(list)
    y=read_wobj(list(i));
    v = getfield(y,'vertices');
    v = reshape(v,3,300);
    [R,T] = icp(base_vertices,v,30);
    f=R*v + repmat(T,1,length(v));
    f=reshape(f,300,3)
    deformed_vertices=f-reshape(base_vertices,300,3);
end
%deformed vertices- vertices after alignment with Base model by using ICP