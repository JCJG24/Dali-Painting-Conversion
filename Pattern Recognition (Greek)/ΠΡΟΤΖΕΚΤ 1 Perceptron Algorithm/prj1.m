%EROTHMA1
[x,c] = ReadLiver(345);

%EROTHMA2
x1 = SeparateClass(x,c,1);
x2 = SeparateClass(x,c,2);

Lr = 0.01
MaxRep = 10000

[Rc,Rep,w] = Perceptron(x1,x2,Lr,MaxRep)

%EROTHMA3
MinError = ((sum(Rep) - sum(Rc))./sum(Rep))*100 


EROTHMA4
Aksiopistia = [];
for Lr = [0.00001 0.0001/2 0.0001 0.001/2 0.001 0.01/2 0.01 0.05 0.1 0.5 1]
    
[Rc,Rep] = Perceptron(x1,x2,Lr,MaxRep);

Aksiopistia = [ Aksiopistia  sum(Rc)/sum(Rep)*100] ;

end

plot(Aksiopistia)


MaxRep = [];
for MaxRep = [100 1000 10000 100000 1000000]
    
[Rc,Rep] = Perceptron(x1,x2,Lr,MaxRep);

Aksiopistia = [ Aksiopistia  sum(Rc)/sum(Rep)*100] ;

end

plot(Aksiopistia)

