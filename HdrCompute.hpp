#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
namespace HdrCompute {

	Eigen::VectorXd weightCurve(int const valueMin, int const valueMax);

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
} 
