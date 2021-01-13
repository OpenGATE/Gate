/*-------------------------------------------------------

List Mode Format 
                        
--  Sigmoid.cc  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2006 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of Sigmoid class:


-------------------------------------------------------*/

#include <stdio.h>
#include <math.h>
#include "lmf.h"
#include "Sigmoid.hh"

Sigmoid::Sigmoid(double alpha, double x0):
  _alpha(alpha),
  _x0(x0)
{
}

Sigmoid::Sigmoid(const Sigmoid & right):
_alpha(right._alpha),
_x0(right._x0)
{
}

double Sigmoid::operator() (double x) const 
{
  double alpha  = _alpha;
  double x0   = _x0;

  return (1. / (1. + exp(-(alpha * (x - x0)))));
}
