/*-------------------------------------------------------

List Mode Format 
                        
--  Gaussian.cc  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2006 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of Gaussian class:


-------------------------------------------------------*/

#include <stdio.h>
#include <math.h>
#include "lmf.h"
#include "Gaussian.hh"

Gaussian::Gaussian(double mean, double sigma):
  _mean(mean),
  _sigma(sigma)
{
}

Gaussian::Gaussian(const Gaussian & right):
_mean(right._mean),
_sigma(right._sigma)
{
}

double Gaussian::operator() (double x) const 
{
  double s   = _sigma;
  double x0  = _mean;
  return (1.0/(sqrt(2*M_PI)*s))*
	  exp(-(x-x0)*(x-x0)/(2.0*s*s));
}

double Gaussian::Shoot(double m, double s)
{
  static double y2;
  static int use_last = 0;
  double x1, x2, w, y1;

  if (use_last) { /* use value from previous call */
    y1 = y2;
    use_last = 0;
  }
  else {
    do {
      x1 = 2.0 * randd() - 1.0;
      x2 = 2.0 * randd() - 1.0;
      w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );
    
    w = sqrt( (-2.0 * log( w ) ) / w );
    y1 = x1 * w;
    y2 = x2 * w;
    use_last = 1;
  }

  return( m + y1 * s );
}
