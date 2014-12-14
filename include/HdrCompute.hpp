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

    //output : lnEi
    Eigen::MatrixXd computeRadianceMap(std::vector <Eigen::MatrixXi > const& images ,
									   Eigen::VectorXd const& g,
									   std::vector <double > const& exposure ,
									   int const valueMin ,
                                       int const valueMax,
                                       double & valueMinIrradiance,
                                       double & valueMaxIrradiance,
                                       bool const outputLinear);

    void toneMappingLinear(char channel, Eigen::MatrixXd const& irradiance, double minIrradiance, double maxIrradiance, kn::ImageRGB8u & res);
    void toneMappingReinhard(char channel, Eigen::MatrixXd const& irradiance, double minIrradiance, double maxIrradiance, kn::ImageRGB8u & res);
    void toneMappingExpClamp(double clamp, char channel, Eigen::MatrixXd const& irradiance, double minIrradiance, double maxIrradiance, kn::ImageRGB8u & res);

    void choiceToneMapping(std::string const & toneMapping, kn::ImageRGB8u & res,
                           char const channel,
                           const Eigen::MatrixXd &responseRecovery,
                           std::vector<Eigen::MatrixXi> const &imagesMatrices,
                           std::vector<double> const& exposures,
                           int const valueMin, int const valueMax);

    void transformImageToMatrix(kn::ImageRGB8u const& im, Eigen::MatrixXi & matrixR,
													  Eigen::MatrixXi & matrixG,  
													  Eigen::MatrixXi & matrixB);

    void transformImageToMatrixGray(kn::ImageRGB8u const& im, Eigen::MatrixXi & matrixGray);

    void handleRGB(std::string const & toneMapping, kn::ImageRGB8u & res, std::vector<Eigen::Vector2i> const& pixels,
                    const std::vector<std::vector<Eigen::MatrixXi> > &imagesMatrices, std::vector<double> const& exposures,
                    int const valueMin, int const valueMax);

    //even if only one channel, we encapsulate gray channel in std::vector to respect generecite in choiceToneMapping
    void handleGray(std::string const & toneMapping, kn::ImageRGB8u & res, std::vector<Eigen::Vector2i> const& pixels,
                     const std::vector<std::vector<Eigen::MatrixXi>> &imGray, std::vector<double> const& exposures,
                     int const valueMin, int const valueMax);

    void tansformChannelsToVector(char const channel, std::vector<int> & channelsOut);
} 
