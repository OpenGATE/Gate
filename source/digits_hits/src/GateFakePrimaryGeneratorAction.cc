/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEFAKEPRIMARYGENERATORACTION_CC
#define GATEFAKEPRIMARYGENERATORACTION_CC

/*
 * \file  GateFakePrimaryGeneratorAction.cc
 * \brief Fake PrimaryGeneratorAction class for development
 */

#include "GateFakePrimaryGeneratorAction.hh"

//GateFakePrimaryGeneratorAction::GateFakePrimaryGeneratorAction( GateFakeDetectorConstruction* GateDC)
GateFakePrimaryGeneratorAction::GateFakePrimaryGeneratorAction( GateDetectorConstruction* GateDC)
  :GateDetector(GateDC)
{
  // G4int n_particle = 1;
  particleGun  = new G4GeneralParticleSource();
  // particleGun  = new G4ParticleGun();


  // default particle kinematic

  /* G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
     G4String particleName;
     G4ParticleDefinition* particle
     = particleTable->FindParticle(particleName="gamma");
     particleGun->SetParticleDefinition(particle);
     particleGun->SetParticleMomentumDirection(G4ThreeVector(1.,0.,0.));
     particleGun->SetParticleEnergy(50.*MeV);
     particleGun->SetParticlePosition(G4ThreeVector(-100*cm,0.*cm,0.*cm));*/

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

GateFakePrimaryGeneratorAction::~GateFakePrimaryGeneratorAction()
{
  delete particleGun;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void GateFakePrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
  //this function is called at the begining of event
  //
  /*G4double x0 = -100*cm;
    G4double y0 = 0.*cm, z0 = 0.*cm;

    particleGun->SetParticlePosition(G4ThreeVector(x0,y0,z0));*/

  particleGun->GeneratePrimaryVertex(anEvent);
}

#endif /* end #define GATEFAKEPRIMARYGENERATORACTION_CC */

//-----------------------------------------------------------------------------
// EOF
//-----------------------------------------------------------------------------
