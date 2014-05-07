/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateInfoParticles
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
 */

#ifndef GATEINFOPARTICLES_HH
#define GATEINFOPARTICLES_HH

#include "globals.hh"

class GateInfoParticles
{
public:
  GateInfoParticles(){}
  ~GateInfoParticles(){}

  G4int mID;
  G4int mParentID;

  G4String mParticleName;
  G4String mProcessName;
  G4String mProcessType;

  G4double mPositionProdX;
  G4double mPositionProdY;
  G4double mPositionProdZ;

  G4int mStepNumber;
  G4double mParentEnergy;
  G4double mEnergy;

};

class GateInfoStep
{
public:
  GateInfoStep(){}
  ~GateInfoStep(){}

  G4ThreeVector mPosition;
  G4double mEnergy;
  G4int nParentID;
};
#endif
