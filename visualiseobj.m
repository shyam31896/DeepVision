basemodel=read_wobj('model1 (2).Obj')
%f-vertices predicted from network
base_vertices=setfield(basemodel,'vertices',f)
write_wobj(base_vertices,'herman.obj')


