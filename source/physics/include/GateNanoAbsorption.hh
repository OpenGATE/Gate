/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateNanoAbsorption_h
#define GateNanoAbsorption_h 1
 
#include "globals.hh"
#include "templates.hh"
#include "Randomize.hh"
#include "G4Step.hh"
#include "G4VDiscreteProcess.hh"
#include "G4DynamicParticle.hh"
#include "G4Material.hh"
#include "G4OpticalPhoton.hh"
#include "G4OpAbsorption.hh"

class GateNanoAbsorption : public G4OpAbsorption
 
{   

 public:
 
         ////////////////////////////////
         // Constructors and Destructor
         ////////////////////////////////
 
         GateNanoAbsorption(const G4String& processName = "NanoAbsorption", G4ProcessType type = fOptical):G4OpAbsorption(processName, type)
{
}

  
     G4double GetMeanFreePath(const G4Track& aTrack,G4double ,G4ForceCondition* ); 
    
};

#endif
