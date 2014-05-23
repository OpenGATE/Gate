/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
  \class  GateHounsfieldDensityTable.cc
  \brief  
  \author david.sarrut@creatis.insa-lyon.fr
*/
 
#include "GateHounsfieldDensityTable.hh"
#include "GateMiscFunctions.hh"
#include "G4SystemOfUnits.hh"

//-----------------------------------------------------------------------------
GateHounsfieldDensityTable::GateHounsfieldDensityTable()
{
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHounsfieldDensityTable::~GateHounsfieldDensityTable()
{
  mH.clear();
  mD.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateHounsfieldDensityTable::GetDensityFromH(double H)
{
  return LinearInterpolation(H, mH, mD);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateHounsfieldDensityTable::FindMaxDensityDifference(double HMin, double HMax) {
  double dMin = GetDensityFromH(HMin);
  double dMax = GetDensityFromH(HMax);

  int i = 0;
  int n = mH.size();
  while (i<n && HMin>mH[i]) i++; //i--;
  int j=0;
  while (j<n && HMax>mH[j]) j++; j--;
  for(int x=i; x<j; x++) {
    // DD(G4BestUnit(mD[x], "Volumic Mass"));
    if (mD[x] < dMin) dMin = mD[x];
    if (mD[x] > dMax) dMax = mD[x];
  }

  return (dMax-dMin);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHounsfieldDensityTable::Read(G4String filename) 
{
  std::ifstream is;
  OpenFileInput(filename, is);
  while (is) {
    skipComment(is);
    double h,d;
    is >> h;
    is >> d;
    if (is) {
      mH.push_back(h);
      mD.push_back(d*g/cm3);
      if (mH.size() > 1) {
	if (h <= mH[mH.size()-2]) {
	  GateError("Error Hounsfield must be in strict ascending order, while I read h="
		    << mH[mH.size()-2] << " and then h=" << h 
		    << " (in file " << filename << ")" << G4endl);
	}
      }
    }
  }
}
//-----------------------------------------------------------------------------
