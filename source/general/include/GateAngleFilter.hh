/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateAngleFilter
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEANGLEFILTER_HH
#define GATEANGLEFILTER_HH


#include "GateVFilter.hh"
#include "GateActorManager.hh"
#include "GateAngleFilterMessenger.hh"


class GateAngleFilter : 
  public GateVFilter
{
public:
  GateAngleFilter(G4String name);
  virtual ~GateAngleFilter(){}

  FCT_FOR_AUTO_CREATOR_FILTER(GateAngleFilter)

  virtual G4bool Accept(const G4Track*);

  void SetMomentum(G4ThreeVector direction);
  // Set acceptable direction.
  //
  void SetAngle(G4double angle);
  // Set acceptable direction.
  //
 
  virtual void show();


private:
  G4ThreeVector mDirection;
  G4double mAngle;
  GateAngleFilterMessenger * pAngleMessenger;

};

MAKE_AUTO_CREATOR_FILTER(angleFilter,GateAngleFilter)


#endif /* end #define GATEANGLEFILTER_HH */
