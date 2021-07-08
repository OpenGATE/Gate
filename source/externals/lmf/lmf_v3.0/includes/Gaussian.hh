#ifndef GAUSSIAN_HH
#define GAUSSIAN_HH

/*********************************************
class Gaussian

The Gaussian distribution is defined by 

f(x) = (1 / (sigma * sqrt(2 *Pi))) * 
       exp(- (((x-m)**2) / (2 sigma**2)))

*********************************************/

class Gaussian {
public:
  // Constructor
  Gaussian(double mean = 0, double sigma = 1);
  // Copy constructor
  Gaussian(const Gaussian &right);
  // Destructor
  ~Gaussian() {};
  // Retreive function value
  // Get the mean of the Gaussian
  double GetMean() { return _mean; }; 
  // Get the sigma of the Gaussian
  double GetSigma() { return _sigma; };
  // Retrieves the Gaussian value of x
  double operator() (double x) const;
  static double Shoot(double m = 0, double s = 1);/* normal random variate generator */
					          /*  mean m, standard deviation s   */
private:
  // Mean
  double _mean;
  // Sigma
  double _sigma;
};
#endif
