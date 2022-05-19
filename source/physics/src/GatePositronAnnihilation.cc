/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GatePositronAnnihilation.hh"
#include "GateSingletonDebugPositronAnnihilation.hh"
#include "G4UnitsTable.hh"
#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

/*
G4VParticleChange* GatePositronAnnihilation::AtRestDoIt(const G4Track& aTrack,
                                                  const G4Step&  aStep)
*/						  
G4VParticleChange* GatePositronAnnihilation::AtRestDoIt(const G4Track& aTrack,
                                                  const G4Step&  )
//
// Performs the e+ e- annihilation when both particles are assumed at rest.
// It generates two back to back photons with energy = electron_mass.
// The angular distribution is isotropic. 
// GEANT4 internal units
//
// Note : Effects due to binding of atomic electrons are negliged.
 
{
   fParticleChange.InitializeForPostStep(aTrack);

   fParticleChange.SetNumberOfSecondaries(2) ;
   
   G4double r  = CLHEP::RandGauss::shoot(0.,0.0011); // hard coded value!
      
   G4double E1 = electron_mass_c2 + r;
   G4double E2 = electron_mass_c2 - r;

   G4double DeltaTeta = 2*r/electron_mass_c2;
   
        G4double cosTeta = G4RandFlat::shoot(-1.0, 1.0);
        G4double sinTeta = sqrt(1.-cosTeta*cosTeta);
        G4double Phi     = twopi * G4UniformRand() ;
	G4double Phi1     = (twopi * G4UniformRand())/2. ;
        G4ThreeVector Direction (sinTeta*cos(Phi), sinTeta*sin(Phi), cosTeta);   
 
     
	G4ThreeVector DirectionPhoton (sin(DeltaTeta)*cos(Phi1),sin(DeltaTeta)*sin(Phi1),cos(DeltaTeta));
	DirectionPhoton.rotateUz(Direction);
	   

	fParticleChange.AddSecondary( new G4DynamicParticle (G4Gamma::Gamma(),
                                                 DirectionPhoton, E1) );
        fParticleChange.AddSecondary( new G4DynamicParticle (G4Gamma::Gamma(),
                                                -Direction, E2) ); 

        fParticleChange.ProposeLocalEnergyDeposit(0.);
       
  // G4double cosdev;
  // G4double dev;

   auto debugPositronAnnihilation = GateSingletonDebugPositronAnnihilation::GetInstance();
   if (debugPositronAnnihilation->GetDebugFlag()){
     G4double tmp;
     std::ofstream out;
     out.open(debugPositronAnnihilation->GetOutputFile(), std::ios::app | std::ios::out | std::ios::binary);
     tmp = DirectionPhoton.angle(- Direction);
     out.write((char*)&tmp, sizeof(double));
     out.close();
   }
  
   // Kill the incident positron 
   //
   fParticleChange.ProposeTrackStatus( fStopAndKill );
      
   return &fParticleChange;

}
