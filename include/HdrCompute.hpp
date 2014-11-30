#include <Eigen/Dense>
#include <vector>
#include "ImageRGB.hpp"

namespace HdrCompute {

	Eigen::VectorXi weightCurve(int const valueMin, int const valueMax);

	Eigen::VectorXd responseRecovery(std::vector <Eigen::MatrixXi> const& images , 
									 std::vector <double> const& exposure ,
									 std::vector <Eigen::Vector2i> const& pixels ,
									 int const valueMin , 
									 int const valueMax ,
									 double const lambda);

	Eigen::MatrixXd computeRadianceMap(std::vector <Eigen::MatrixXi > const& images ,
									   Eigen::VectorXd const& g,
									   std::vector <double > const& exposure ,
									   int const valueMin ,
									   int const valueMax);

	Eigen::MatrixXi toneMapping(char channel, int const width, int const height, Eigen::MatrixXd const& irradiance, kn::ImageRGB8u & res);

	void transformImageToMatrix(kn::ImageRGB8u const& im, Eigen::MatrixXi & matrixR,
													  Eigen::MatrixXi & matrixG,  
													  Eigen::MatrixXi & matrixB);

	void transformImageToMatrixGray(kn::ImageRGB8u const& im, Eigen::MatrixXi & matrixGray);

	void handleRGB( kn::ImageRGB8u & res, std::vector<Eigen::Vector2i> const& pixels, 
					std::vector<Eigen::MatrixXi> const& imR, std::vector<Eigen::MatrixXi> const& imG,
					std::vector<Eigen::MatrixXi> const& imB, std::vector<double> const& exposures, 
					int const valueMin, int const valueMax);
	void handleGray( kn::ImageRGB8u & res, std::vector<Eigen::Vector2i> const& pixels,
					 std::vector<Eigen::MatrixXi> const& imGray, std::vector<double> const& exposures, 
					int const valueMin, int const valueMax);
} 
