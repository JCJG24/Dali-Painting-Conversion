function [Rc,Rep,Weights] = Perceptron(x1,x2,Lr,MaxRep)
%#
%#  [Rc,Rep] = Perceptron(x,c,Lr)
%#
%#  Input
%#      x1: Pattern Vectors for the first class
%#      x2: Pattern Vectors for the second class
%#      Lr: Learning rate
%#      MaxRep: Maximum repetitions
%#  Output
%#      Rc: Correct classification rate using the C-method
%#      Rep: Pattern vectors on each class
%#
[rows1 columns1] = size(x1);
[rows2 columns2] = size(x2);
NumOfP1 = columns1 ;
NumOfP2 = columns2 ;
Weights = 2 * rand(rows1+1,1) - 1 ;
Rep = [NumOfP1,NumOfP2] ;
TotPat = sum(Rep) ;
Rc = zeros(2,1) ;
if ( size(x1,1) ~= size(x2,1) )
    printf( 'Error in vectors x1, x2\n' ) ;
end

%#
%#  C-Error
%#

for j = 1:MaxRep
	if ( rand() > 0.5 )
		k = floor(rand() * NumOfP1 + 1) ; 
		Pat = [x1(:,k);1] ;
		Score = Weights' * Pat ;
		if ( Score < 0 )
			Weights = Weights + Lr * Pat ;
		end
	else
      k = floor(rand() * NumOfP2 + 1) ;
      Pat = [x2(:,k);1] ;
      Score = Weights' * Pat ;
      if ( Score >= 0 )
         Weights = Weights - Lr * Pat ;
      end
	end
	Rc = zeros(2,1) ;
	for i=1:NumOfP1
		if ( Weights' * [x1(:,i);1] >= 0 )
			Rc(1) = Rc(1) + 1 ;
		end
	end
   for i=1:NumOfP2
      if ( Weights' * [x2(:,i);1] < 0 )
         Rc(2) = Rc(2) + 1 ;
      end
   end
k=sum(Rc); 
fprintf( '%8d', k ) ;
	if ( sum(Rc) == TotPat )
		break ;
	end
end
fprintf( '\n' ) ;
