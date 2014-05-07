/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
  \class  GateHounsfieldDensityTable.hh
  \brief  
  \author david.sarrut@creatis.insa-lyon.fr
*/
 
#ifndef __GateHounsfieldDensityTable__hh__
#define __GateHounsfieldDensityTable__hh__

#include "GateHounsfieldMaterialProperties.hh"
#include "GateMiscFunctions.hh"
#include "G4UnitsTable.hh"

class GateHounsfieldDensityTable
{
public:
  GateHounsfieldDensityTable();
  ~GateHounsfieldDensityTable();

  double GetDensityFromH(double H);
  double FindMaxDensityDifference(double HMin, double HMax);
  void Read(G4String filename);

protected:
  std::vector<double> mH;
  std::vector<double> mD;

};
#endif
