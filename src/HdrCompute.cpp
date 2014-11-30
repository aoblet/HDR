#include "HdrCompute.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

Eigen::VectorXi HdrCompute::weightCurve(int const valueMin, int const valueMax){
	Eigen::VectorXi weight = Eigen::VectorXi::Ones(256);
	for(int i=valueMin; i<=valueMax; ++i){
		if(i > 0.5*	(valueMin + valueMax))
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
	if(images.size() <= 1)
		throw std::invalid_argument("responseRecovery: Images vectors must be greater than 1");

	uint Z = 256;
	int NP = pixels.size()*images.size();

	if( (int)(NP - pixels.size()) < valueMax - valueMin)
		throw std::invalid_argument("responseRecovery: No solution if N(P-1) < Zmax - Zmin");

	Eigen::MatrixXd A = Eigen::MatrixXd::Zero( NP+Z+1, Z+pixels.size() );
	Eigen::VectorXd x(Z + pixels.size()), b = Eigen::VectorXd::Zero(A.rows());

	Eigen::VectorXi	w(HdrCompute::weightCurve(valueMin,valueMax));

	int currentLine =0;
	int valPix;

	//fill A and b
	for( uint i=0; i<pixels.size(); ++i ){
		for( uint j=0; j< images.size(); ++j){
			valPix = images[j](pixels[i](1),pixels[i](0));

			A(currentLine, valPix) = w(valPix);			//according g(Z)
			A(currentLine, Z+i) = -w(valPix);			//according Ei
			b(currentLine) = w(valPix)*log(exposure[j]);
			++currentLine;
		}
	}

	//scale factor
	A(currentLine++, Z/2) = 1;
	b(currentLine-1) = 1;

	//fill for smoothness	
	A(currentLine, 0) = w(0)*-2*lambda;
	A(currentLine, 1) = w(0)*lambda;
	++currentLine;

	for( uint i=1; i<Z-1; ++i){
		A(currentLine,i-1) = w(i)*lambda;
		A(currentLine,i)   = w(i)*-2*lambda;
		A(currentLine,i+1) = w(i)*lambda;
		++currentLine;
	}

	A(currentLine, Z-2) = w(Z-1)*lambda;
	A(currentLine, Z-1) = w(Z-1)*-2*lambda;

	//solve according least squares
	x = (A.transpose()*A).inverse()*A.transpose()*b;
	return x.block(0,0,Z,1);
}

Eigen::MatrixXd HdrCompute::computeRadianceMap(std::vector <Eigen::MatrixXi > const& images ,
							Eigen::VectorXd const& g,
							std::vector <double > const& exposure ,
							int const valueMin ,
							int const valueMax){
	if(images.size() <= 1)
		throw std::invalid_argument("computeRadianceMap: images size invalid");

	Eigen::MatrixXd irradiance = Eigen::MatrixXd::Zero(images[0].rows(), images[0].cols());
	Eigen::VectorXi w(HdrCompute::weightCurve(valueMin,valueMax));

	uint width=images[0].cols(), height=images[0].rows(), valPix, xImage, yImage;
	double tmpDenominator;

	for(uint i =0; i< width*height; ++i){
		tmpDenominator=0;
		
		xImage = i%width; //images pixels line to rect
		yImage = i/width;

		for(uint j =0; j< images.size(); ++j){
			valPix = images[j](yImage, xImage);
			irradiance(yImage,xImage) += (w(valPix)*g(valPix))-(w(valPix)*log(exposure[j]));
			tmpDenominator += w(valPix);
		}
		tmpDenominator = tmpDenominator == 0 ? 1 : tmpDenominator;
		irradiance(yImage, xImage) /= tmpDenominator;
		irradiance(yImage, xImage) = exp(irradiance(yImage,xImage));

	}

	return irradiance;
}

//devebec version: irradiance storage = Ei != lnEi with range radiance acceptation
Eigen::MatrixXi HdrCompute::toneMapping(char channel, int const width, int const height, Eigen::MatrixXd const& irradiance, kn::ImageRGB8u & res){
		
	double 	minIrradiance = irradiance.minCoeff(), 
			maxIrradiance = (irradiance.maxCoeff() - minIrradiance)*0.02;

	std::cout << "Irradiance min/max (toneMapping)\n" <<minIrradiance << " " << maxIrradiance << std::endl;

	Eigen::MatrixXi resCompute = Eigen::MatrixXi::Zero(height,width);
	double  tmpValPixel;

	for(unsigned i = 0; i < irradiance.rows(); ++i) {
		for(unsigned j = 0; j < irradiance.cols(); ++j) {
			if(irradiance(i,j) - minIrradiance > maxIrradiance)
				tmpValPixel = (int)round( (( maxIrradiance * 255 )/ maxIrradiance) );
			else
				tmpValPixel = (int)round( (((irradiance(i,j) - minIrradiance) * 255 )/ maxIrradiance));

			//tmpValPixel = ((irradiance(i,j)-minIrradiance) * 255)/ maxIrradiance; //classic toneMapping with lnEi
			resCompute(i,j) = round(tmpValPixel);
		}						  
	}

	
	std::vector<int> channels;

	if(channel == 'A'){
		channels.push_back(0);
		channels.push_back(1);
		channels.push_back(2);
	}
	else if(channel == 'R')
		channels.push_back(0);
	else if(channel == 'G')
		channels.push_back(1);
	else if(channel == 'B')
		channels.push_back(2);
	else
		throw std::invalid_argument("Invalid channel in tone mapping");

	uint channelInd,j;
	for(uint i=0; i< (uint)height; ++i){
		for(j=0; j<(uint)width; ++j){
			for(channelInd = 0; channelInd < channels.size(); ++channelInd){
				res(j,i)[channels[channelInd]] = resCompute(i,j);
				//if(resCompute(i,j) < 2) //fix some bugs
				//	res(j,i)[channels[channelInd]] = 255;
			}
		}
	}
	
	return resCompute;
}

void HdrCompute::transformImageToMatrix(kn::ImageRGB8u const& im, Eigen::MatrixXi & matrixR,
																  Eigen::MatrixXi & matrixG, 
																  Eigen::MatrixXi & matrixB){
	for(unsigned int i=0; i<im.height(); ++i){
		for(unsigned int j=0; j<im.width();++j){
			matrixR(i,j) = im(j,i)[0];
			matrixG(i,j) = im(j,i)[1];
			matrixB(i,j) = im(j,i)[2];
		}
	}
}

void HdrCompute::transformImageToMatrixGray(kn::ImageRGB8u const& im, Eigen::MatrixXi & matrixGray){
	for(unsigned int i=0; i<im.height(); ++i){
		for(unsigned int j=0; j<im.width();++j){
			matrixGray(i,j) = 0.2126 * im(j,i)[0] + 0.7152 * im(j,i)[1] + 0.0722 * im(j,i)[2];
		}
	}
}

void HdrCompute::handleRGB( kn::ImageRGB8u & res, std::vector<Eigen::Vector2i> const& pixels, 
							std::vector<Eigen::MatrixXi> const& imR,std::vector<Eigen::MatrixXi> const& imG, 
							std::vector<Eigen::MatrixXi> const& imB, std::vector<double> const& exposures,
							int const valueMin, int const valueMax){

	if(imR.size() < 2 || imG.size() < 2 || imB.size() < 2)
		throw std::invalid_argument("handleRGB: incorrect size for input images");

	std::numeric_limits<short> limit;

	uint widthImage = imR[0].cols(), heightImage=imR[0].rows();
	Eigen::VectorXd responseRecoveryR = HdrCompute::responseRecovery(imR, exposures, pixels, valueMin, valueMax, limit.max());
	Eigen::VectorXd responseRecoveryG = HdrCompute::responseRecovery(imG, exposures, pixels, valueMin, valueMax, limit.max());
	Eigen::VectorXd responseRecoveryB = HdrCompute::responseRecovery(imB, exposures, pixels, valueMin, valueMax, limit.max());
  	
  	Eigen::MatrixXd computeRadianceMapR = HdrCompute::computeRadianceMap(imR, responseRecoveryR, exposures, valueMin, valueMax );
  	Eigen::MatrixXd computeRadianceMapG = HdrCompute::computeRadianceMap(imG, responseRecoveryG, exposures, valueMin, valueMax );
  	Eigen::MatrixXd computeRadianceMapB = HdrCompute::computeRadianceMap(imB, responseRecoveryB, exposures, valueMin, valueMax );

  	HdrCompute::toneMapping('R',widthImage, heightImage, computeRadianceMapR, res);
  	HdrCompute::toneMapping('G',widthImage, heightImage, computeRadianceMapG, res);
  	HdrCompute::toneMapping('B',widthImage, heightImage, computeRadianceMapB, res);
} 

void handleGray( kn::ImageRGB8u & res, std::vector<Eigen::Vector2i> const& pixels,
				 std::vector<Eigen::MatrixXi> const& imGray, std::vector<double> const& exposures, 
				int const valueMin, int const valueMax){

	if(imGray.size() < 2)
		throw std::invalid_argument("handleGray: incorrect size for input images");

	uint widthImage = imGray[0].cols(), heightImage=imGray[0].rows();
  	Eigen::VectorXd responseRecoveryGray = HdrCompute::responseRecovery(imGray, exposures, pixels, valueMin, valueMax, std::numeric_limits<short>().max() );
  	Eigen::MatrixXd computeRadianceMapGray = HdrCompute::computeRadianceMap(imGray, responseRecoveryGray, exposures, valueMin, valueMax );
  	HdrCompute::toneMapping('A',widthImage, heightImage, computeRadianceMapGray, res);
}