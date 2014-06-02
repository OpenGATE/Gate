/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEFAKEPRIMARYGENERATORACTION_HH
#define GATEFAKEPRIMARYGENERATORACTION_HH


/*
 * \file  GateFakePrimaryGeneratorAction.hh
 * \brief Fake PrimaryGeneratorAction class for development
 */


#include "G4VUserPrimaryGeneratorAction.hh"
//#include "GateFakeDetectorConstruction.hh"
#include "GateDetectorConstruction.hh"

#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4GeneralParticleSource.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "globals.hh"

class GateFakePrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
  //    GateFakePrimaryGeneratorAction(GateFakeDetectorConstruction*);
  GateFakePrimaryGeneratorAction(GateDetectorConstruction*);
  ~GateFakePrimaryGeneratorAction();

public:
  void GeneratePrimaries(G4Event*);

private:
  G4GeneralParticleSource *                particleGun;	  //pointer a to G4  class

};


#endif /* end #define GATEFAKEPRIMARYGENERATORACTION_HH */

//-----------------------------------------------------------------------------
// EOF
//-----------------------------------------------------------------------------
