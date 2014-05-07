/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateUserLimits
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEUSERLIMITS_HH
#define GATEUSERLIMITS_HH

#include "globals.hh"

class GateUserLimits
{

public:
  GateUserLimits(){Clear();}
  ~GateUserLimits(){}

  void SetMaxStepSize(double value){mMaxStepSize=value;}
  void SetMaxTrackLength(double value){mMaxTrackLength=value;}
  void SetMaxToF(double value){mMaxToF=value;}
  void SetMinKineticEnergy(double value){mMinKineticEnergy=value;}
  void SetMinRemainingRange(double value){mMinRemainingRange=value;}

  double GetMaxStepSize(){return mMaxStepSize;}
  double GetMaxTrackLength(){return mMaxTrackLength;}
  double GetMaxToF(){return mMaxToF;}
  double GetMinKineticEnergy(){return mMinKineticEnergy;}
  double GetMinRemainingRange(){return mMinRemainingRange;}

  void Clear(){mMaxStepSize=-1.; mMaxTrackLength=-1.; mMaxToF=-1.; mMinKineticEnergy=-1.; mMinRemainingRange=-1.; }

private:
  double mMaxStepSize;
  double mMaxTrackLength;
  double mMaxToF;
  double mMinKineticEnergy;
  double mMinRemainingRange;

};


#endif /* end #define GATEUSERLIMITS_HH */
