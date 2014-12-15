#include "HdrCompute.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <fstream>

Eigen::VectorXi HdrCompute::weightCurve(int const valueMin, int const valueMax){
    if(valueMin < 0 || valueMax > 255)
        throw std::invalid_argument("valueMin-valueMax incorrect for weightCurve");

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
    if(images.size() < 2)
        throw std::invalid_argument("responseRecovery: Images vectors must be greater than 1");

    uint NP = pixels.size()*images.size(), Z = 256;

    if( (int)(NP - pixels.size()) < valueMax - valueMin)
        throw std::invalid_argument("responseRecovery: No solution if N(P-1) < Zmax - Zmin");

    Eigen::VectorXi	w(HdrCompute::weightCurve(valueMin,valueMax));
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero( NP+Z-1, Z+pixels.size() );
    Eigen::VectorXd x(Z + pixels.size()), b = Eigen::VectorXd::Zero(A.rows());

    int currentLine =0, valPix;
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
    A(currentLine, 128) =1;
    b(currentLine) = 1;
    ++currentLine;

    //fill for smoothness
    for( uint i=1; i<Z-1; ++i){
        A(currentLine,i-1) = w(i)*lambda;
        A(currentLine,i)   = w(i)*-2*lambda;
        A(currentLine,i+1) = w(i)*lambda;
        ++currentLine;
    }

    //solve according least squares
    x = (A.transpose()*A).inverse()*A.transpose()*b;
    return x.head(256);
}

Eigen::MatrixXd HdrCompute::computeRadianceMap( std::vector <Eigen::MatrixXi > const& images ,
                                                Eigen::VectorXd const& g,std::vector <double> const& exposure ,
                                                int const valueMin ,int const valueMax,
                                                double &valueMinIrradiance,double &valueMaxIrradiance,
                                                const bool outputLinear){
	if(images.size() <= 1)
		throw std::invalid_argument("computeRadianceMap: images size invalid");

    uint width=images[0].cols(), height=images[0].rows(), valPix;
    double tmpDenominator;
    valueMaxIrradiance = (valueMinIrradiance = std::numeric_limits<double>::infinity());

    Eigen::MatrixXd irradiance = Eigen::MatrixXd::Zero(height, width);
	Eigen::VectorXi w(HdrCompute::weightCurve(valueMin,valueMax));

    for(uint xImage=0; xImage<width; ++xImage){
        for(uint yImage=0; yImage<height; ++yImage){
            tmpDenominator=0;
            //we also include ponderation here
            for(uint j =0; j< images.size(); ++j){
                valPix = images[j](yImage, xImage);
                irradiance(yImage,xImage) += w(valPix)*(g(valPix)-log(exposure[j]));
                tmpDenominator += w(valPix);
            }
            irradiance(yImage, xImage) /= tmpDenominator;
            irradiance(yImage, xImage) = outputLinear ? irradiance(yImage,xImage) : exp(irradiance(yImage,xImage));

            //compute properly min max
            if(!isnan(irradiance(yImage, xImage))){
                if(valueMaxIrradiance == std::numeric_limits<double>::infinity() || irradiance(yImage, xImage) > valueMaxIrradiance)
                    valueMaxIrradiance = irradiance(yImage, xImage);
                if(valueMinIrradiance == std::numeric_limits<double>::infinity() || irradiance(yImage, xImage) < valueMinIrradiance)
                    valueMinIrradiance = irradiance(yImage, xImage);
            }
        }
	}
    return irradiance;
}

void HdrCompute::toneMappingLinear(char channel, Eigen::MatrixXd const& irradiance,  double minIrradiance,  double maxIrradiance, kn::ImageRGB8u & res){
    std::cout << "Irradiance min/max (toneMappingLinear)\n" <<minIrradiance << " " << maxIrradiance << std::endl;

    maxIrradiance -= minIrradiance;

    std::vector<int> channels;
    uint channelInd,tmpValPixel;
    HdrCompute::tansformChannelsToVector(channel,channels);

    //compute & fill pixel
    for(uint i = 0; i < irradiance.rows(); ++i) {
        for(uint j = 0; j < irradiance.cols(); ++j) {
            tmpValPixel = abs(round(((irradiance(i,j)-minIrradiance) * 255.0)/ maxIrradiance)); //classic toneMappingLinear with lnEi
            if(isnan(irradiance(i,j)) || tmpValPixel > 255)
                tmpValPixel = 255;

            //fill channel(s) desired
            for(channelInd = 0; channelInd < channels.size(); ++channelInd){
                res(j,i)[channels[channelInd]] = tmpValPixel;
            }
		}						  
    }
}

void HdrCompute::toneMappingExpClamp(double clamp, char channel, Eigen::MatrixXd const& irradiance, double minIrradiance, double maxIrradiance, kn::ImageRGB8u & res){
    std::cout << "Irradiance min/max (toneMappingExpClamp)\n" <<minIrradiance << " " << maxIrradiance << std::endl;

    maxIrradiance -= minIrradiance;
    maxIrradiance *= clamp;

    std::vector<int> channels;
    uint channelInd,tmpValPixel;
    HdrCompute::tansformChannelsToVector(channel,channels);

    //compute & fill pixel
    for(uint i = 0; i < irradiance.rows(); ++i) {
        for(uint j = 0; j < irradiance.cols(); ++j) {
            tmpValPixel = abs(round(((irradiance(i,j)-minIrradiance) * 255.0)/ maxIrradiance));
            if(isnan(irradiance(i,j)) || (irradiance(i,j) - minIrradiance) > maxIrradiance)
                tmpValPixel = 255;

            //fill channel(s) desired
            for(channelInd = 0; channelInd < channels.size(); ++channelInd){
                res(j,i)[channels[channelInd]] = tmpValPixel;
            }
        }
    }
}

void HdrCompute::toneMappingReinhard(char channel, Eigen::MatrixXd const& irradiance,  double minIrradiance,  double maxIrradiance, kn::ImageRGB8u & res){
    std::cout << "Irradiance min/max (toneMappingReinhard)\n" <<minIrradiance << " " << maxIrradiance << std::endl;

    std::vector<int> channels;
    uint channelInd,tmpValPixel;
    HdrCompute::tansformChannelsToVector(channel,channels);

    //log average
    double lwB=0, a, ltmp;
    for(uint i = 0; i < irradiance.rows(); ++i) {
        for(uint j = 0; j < irradiance.cols(); ++j) {
            if(!isnan(irradiance(i,j)))
                lwB += log(irradiance(i,j)-minIrradiance+0.01);
        }
    }

    //reinhard formula: curve response: log hight concentrated at min
    lwB = exp(lwB/(irradiance.cols()*irradiance.rows()));
    for(uint i = 0; i < irradiance.rows(); ++i) {
        for(uint j = 0; j < irradiance.cols(); ++j) {
            a = 0.75;
            ltmp =(a/ lwB)*(irradiance(i,j)-minIrradiance);
            tmpValPixel = ((ltmp*(1+(ltmp/(50))))/(1+ltmp))*255;

            if(isnan(irradiance(i,j)) || tmpValPixel > 255)
                tmpValPixel = 255;

            for(channelInd = 0; channelInd < channels.size(); ++channelInd){
                res(j,i)[channels[channelInd]] = tmpValPixel;
            }
        }
    }
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

void HdrCompute::handleRGB( const std::string &toneMapping, kn::ImageRGB8u & res, std::vector<Eigen::Vector2i> const& pixels,
                            std::vector<std::vector< Eigen::MatrixXi>> const& imagesMatrices , std::vector<double> const& exposures,
                            int const valueMin, int const valueMax){

    if(imagesMatrices.size() < 3|| imagesMatrices[0].size() <2 || imagesMatrices[1].size() < 2 || imagesMatrices[2].size() <2)
		throw std::invalid_argument("handleRGB: incorrect size for input images");

    //0:R, 1:G, 2:B
    Eigen::VectorXd responseRecoverieTmp;
    char channel = 'R';

    for(int i=0; i<3; ++i){
        responseRecoverieTmp = HdrCompute::responseRecovery(imagesMatrices[i], exposures, pixels, valueMin, valueMax , 20);
        HdrCompute::choiceToneMapping( toneMapping, res, channel, responseRecoverieTmp, imagesMatrices[i], exposures,valueMin,valueMax);
        channel = channel == 'R' ? 'G' : 'B';
    }
} 

void HdrCompute::handleGray(const std::string &toneMapping, kn::ImageRGB8u & res, std::vector<Eigen::Vector2i> const& pixels,
                             std::vector<std::vector<Eigen::MatrixXi>> const& imGray, std::vector<double> const& exposures,
                             int const valueMin, int const valueMax){

    if(imGray.size() == 0 || imGray[0].size() < 2)
		throw std::invalid_argument("handleGray: incorrect size for input images");

    Eigen::VectorXd responseRecoveryGray = HdrCompute::responseRecovery(imGray[0], exposures, pixels, valueMin, valueMax, 20 );
    HdrCompute::choiceToneMapping( toneMapping, res, 'A', responseRecoveryGray, imGray[0], exposures,valueMin,valueMax);
}

void HdrCompute::tansformChannelsToVector(char const channel, std::vector<int> & channelsOut){
    channelsOut.clear();
    if(channel == 'A'){
        channelsOut.push_back(0);
        channelsOut.push_back(1);
        channelsOut.push_back(2);
    }
    else if(channel == 'R')
        channelsOut.push_back(0);
    else if(channel == 'G')
        channelsOut.push_back(1);
    else if(channel == 'B')
        channelsOut.push_back(2);
    else
        throw std::invalid_argument("Invalid channel in tone mapping");
}

void HdrCompute::choiceToneMapping( std::string const & toneMapping, kn::ImageRGB8u & res,
                                    char const channel,
                                    Eigen::MatrixXd const & responseRecovery,
                                    std::vector<Eigen::MatrixXi> const& imagesMatrices,
                                    std::vector<double> const& exposures,
                                    int const valueMin, int const valueMax){

    double minIrradiance,maxIrradiance; //modified in computeRadiance to serve toneMapping
    Eigen::MatrixXd computeRadianceMap;

    if(toneMapping == "reinhard"){
        computeRadianceMap = HdrCompute::computeRadianceMap(imagesMatrices, responseRecovery, exposures, valueMin, valueMax,minIrradiance, maxIrradiance,false );
        HdrCompute::toneMappingReinhard(channel, computeRadianceMap,minIrradiance, maxIrradiance, res);
    }
    else if(toneMapping == "expClamp"){
        computeRadianceMap = HdrCompute::computeRadianceMap(imagesMatrices, responseRecovery, exposures, valueMin, valueMax,minIrradiance, maxIrradiance,false );
        HdrCompute::toneMappingExpClamp(0.02,channel, computeRadianceMap,minIrradiance, maxIrradiance, res);
    }
    else if(toneMapping == "linear"){
        //linear on lnEi
        computeRadianceMap = HdrCompute::computeRadianceMap(imagesMatrices, responseRecovery, exposures, valueMin, valueMax,minIrradiance, maxIrradiance,true );
        HdrCompute::toneMappingLinear(channel, computeRadianceMap,minIrradiance, maxIrradiance, res);
    }
    else
        throw std::invalid_argument("invalid type of toneMapping");
}
