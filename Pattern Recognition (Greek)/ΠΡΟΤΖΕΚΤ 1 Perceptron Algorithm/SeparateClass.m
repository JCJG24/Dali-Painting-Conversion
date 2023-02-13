function [Vec] = SeparateClass(x,c,Class)
%#
%#  [Vec] = SeparateClass(x,c,Class) 
%#
%#  Input
%#      x: Pattern Vectors
%#      c: Class for each Pattern Vector
%#      Class: Number of Class
%#  Output
%#      Vec: Pattern Vectors of "Class" 
%#
[rows columns]=size(x);
NumOfP = columns ;
Vec = zeros( rows, 0 ) ;

for i = 1:NumOfP
	if ( c(i) == Class )
		Vec = [ Vec, x(:,i) ] ;
	end
end
