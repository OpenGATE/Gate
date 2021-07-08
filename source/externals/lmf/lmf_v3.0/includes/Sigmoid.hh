#ifndef SIGMOID_HH
#define SIGMOID_HH

/*********************************************
class Sigmoid

The sigmoid function is defined by:

f(x) = 1 / (1 + Exp(alpha (x - x0)))

the sigmoid function satisfies this property:

f'(x) = alpha * f(x) * (1 - f(x))

*********************************************/


class Sigmoid {
public:
  // Constructor
  Sigmoid(double alpha = 1, double x0 = 0);
  // Copy constructor
  Sigmoid(const Sigmoid &right);
  // Destructor
  ~Sigmoid() {};
  // Retreive function value
  // Get the alpha param of the Sigmoid
  double GetAlpha() { return _alpha; }; 
  // Get the x0 param of the Sigmoid
  double GetX0() { return _x0; }; 
  // Retrieves the Sigmoid value of x
  double operator() (double x) const;
private:
  // alpha
  double _alpha;
  // x0
  double _x0;
};
#endif
