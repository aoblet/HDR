#include "HdrCompute.hpp"
#include <cmath>

Eigen::VectorXd HdrCompute::weightCurve(int const valueMin, int const valueMax){
	Eigen::VectorXd weight(valueMax - valueMin +1);

	for(int i=valueMin; i<=valueMax; ++i){
		if(i > (valueMin + valueMax)/2)
			weight(i) = valueMax - i;
		else
			weight(i) = i - valueMin;
	}
	return weight;
}

Eigen::VectorXd HdrCompute::responseRecovery(std::vector <Eigen::MatrixXi> const& images , 
									 std::vector <double> const& exposure ,
									 std::vector <Eigen::Vector2i> const& pixels ,
									 int const valueMin , 
									 int const valueMax ,
									 double const lambda){

	unsigned int Z = valueMax - valueMin + 1;
	int NP = pixels.size()*images.size();

	Eigen::MatrixXd A = Eigen::MatrixXd::Zero( NP + Z, Z + pixels.size() );
	Eigen::VectorXd x(Z + pixels.size()), b = Eigen::VectorXd::Zero(NP + Z), w(HdrCompute::weightCurve(valueMin,valueMax));

	int currentLine =0;
	int valPix;

	//fill A and b
	for( unsigned int i=0; i<pixels.size(); ++i ){
		for( unsigned int j=0; j< images.size(); ++j){
			valPix = images[j](pixels[i](0),pixels[i](1));

			A(currentLine, valPix) = w(valPix);			//according g(Z)
			A(currentLine, i + Z) = -w(valPix);			//according Ei
			b(currentLine) = w(valPix)*log(exposure[j]);
			++currentLine;
		}
	}

	//fill for smoothness
	A(currentLine, 0) = -2*lambda;
	A(currentLine, 1) = lambda;
	++currentLine;

	for( unsigned int i=1; i<Z-1; ++i){
		A(currentLine,i-1) = lambda;
		A(currentLine,i)   = -2*lambda;
		A(currentLine,i+1) = lambda;
		++currentLine;
	}

	A(currentLine, Z-2) = lambda;
	A(currentLine, Z-1) = -2*lambda;

	//solve according least squares
	x = (A.transpose()*A).inverse()*A.transpose()*b;
	x((valueMax-valueMin)/2) = 1;	//stabilize
	x.resize(Z);
	return x;
}