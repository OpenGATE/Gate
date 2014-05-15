/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePositronAnnihilation_h
#define GatePositronAnnihilation_h 1

#include "G4ios.hh" 
#include "globals.hh"
#include "Randomize.hh" 
#include "G4VRestDiscreteProcess.hh"
#include "G4PhysicsTable.hh"
#include "G4PhysicsLogVector.hh" 
#include "G4ElementTable.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"
#include "G4Step.hh"
#include "G4eplusAnnihilation.hh"

class GatePositronAnnihilation : public G4eplusAnnihilation
 
{    
  public:
     
     G4VParticleChange* AtRestDoIt(const G4Track& aTrack,
                                  const G4Step& aStep); 
};

#endif
 
