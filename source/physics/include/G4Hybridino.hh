/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*!
  \class  G4Hybridino
  \author francois.smekens@creatis.insa-lyon.fr
 */

#ifndef G4HYBRIDINO_HH
#define G4HYBRIDINO_HH

#include "globals.hh"
#include "G4ios.hh"
#include "G4ParticleDefinition.hh"

//------------------------------------------------------------------------

class G4Hybridino : public G4ParticleDefinition
{
private:
	static G4Hybridino* theInstance;
	G4Hybridino(){}
	~G4Hybridino(){}

public:
	static G4Hybridino* Definition();
	static G4Hybridino* HybridinoDefinition();
	static G4Hybridino* Hybridino();
};

#endif
