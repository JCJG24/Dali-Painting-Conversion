[x,c] = ReadDiabetes(768)

x1 = SeparateClass(x,c,1);
x2 = SeparateClass(x,c,2);

[Rc,Rep] = HoKa(x1,x2,0.01,10000);

Error=(1-(sum(Rc)./sum(Rep)))*100;

Aks = [];
for Lr =0.1:0.1:1
[Rc,Rep] = HoKa(x1,x2,Lr,10000)
Aks = [ Aks  sum(Rc)/sum(Rep)*100] 
end
plot(Aks)

%#Aks = [];
%#MaxRep = [];
%#for MaxRep=100:100:10000
%#[Rc,Rep] = HoKa(x1,x2,0.01,MaxRep);
%#end
%#plot(MaxRep)