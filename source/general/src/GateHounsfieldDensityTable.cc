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
  for (GateHDensTableVec::iterator it=HDensTableVec.begin(); it!=HDensTableVec.end(); )
	  it=HDensTableVec.erase(it);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateHounsfieldDensityTable::GetDensityFromH(double H)
{
	GateHDensTableVec::iterator it=HDensTableVec.begin();
	while(it->mH < H && it!=HDensTableVec.end()) it++;
	if(it==HDensTableVec.begin()) return it->mD;//first value
	if(it==HDensTableVec.end())
	{
		it--;
		return it->mD; //last value
	}
	GateHDensTable prev = HDensTableVec[it - HDensTableVec.begin() - 1];
	return ((H-prev.mH)/(it->mH-prev.mH)) * (it->mD-prev.mD) + prev.mD;

	//return LinearInterpolation(H, mH, mD);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateHounsfieldDensityTable::FindMaxDensityDifference(double HMin, double HMax) {
  double dMin = GetDensityFromH(HMin);
  double dMax = GetDensityFromH(HMax);

  GateHDensTableVec::iterator it=HDensTableVec.begin();
  while (it->mH < HMin && it!=HDensTableVec.end()) it++;
  for ( ; it!=HDensTableVec.end() && it->mH < HMax; it++)
  {
	  if (it->mD < dMin) dMin = it->mD;
	  if (it->mD > dMax) dMax = it->mD;
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
    if (is)
    {
    	GateHDensTable MM;
    	MM.mD=d*g/cm3;
    	MM.mH=h;
    	HDensTableVec.push_back(MM);
      if (HDensTableVec.size() > 1) {
	if (h <= HDensTableVec[HDensTableVec.size()-2].mH) {
	  GateError("Error Hounsfield must be in strict ascending order, while I read h="
		    << HDensTableVec[HDensTableVec.size()-2].mH << " and then h=" << h
		    << " (in file " << filename << ")\n");
	}
      }
    }
  }
}
//-----------------------------------------------------------------------------
