/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateGammaSourceModelSingleGamma.hh"
#include "TLorentzVector.h"
#include "G4Electron.hh"
#include "G4PhysicalConstants.hh"
#include "GateConstants.hh"
#include <cmath>

GateGammaSourceModelSingleGamma* GateGammaSourceModelSingleGamma::ptrJPETSingleGammaModel = 0;
GateGammaSourceModelSingleGamma::GateGammaSourceModelSingleGamma()
{
 SetParticlesNumber(1);
 SetGammaSourceModel( GateGammaModelPrimaryParticleInformation::GammaSourceModel::Single );
 GateExtendedVSourceManager::GetInstance()->AddGammaSourceModel( this );
}

GateGammaSourceModelSingleGamma::~GateGammaSourceModelSingleGamma() {}

void GateGammaSourceModelSingleGamma::GetGammaParticles( std::vector<G4PrimaryParticle*>& particles )
{
 Double_t x = 0, y = 0, z = 0;
 GetRandomGenerator()->Sphere( x, y, z, 1.0 );

 G4ThreeVector momentum_direction( x, y, z );
 G4double kinetic_energy = GetPromptGammaEnergy();
 G4ThreeVector gamma_polarization = GetPolarization( momentum_direction );

 particles[ 0 ]->SetMomentumDirection( momentum_direction );
 particles[ 0 ]->SetMass( 0.0 );
 particles[ 0 ]->SetKineticEnergy( kinetic_energy );
 particles[ 0 ]->SetPolarization( gamma_polarization );
 particles[ 0 ]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaSingle, gamma_polarization ) );
}

G4String GateGammaSourceModelSingleGamma::GetModelName() const { return "singleGamma"; }

GateGammaSourceModelSingleGamma* GateGammaSourceModelSingleGamma::GetInstance()
{
 if( !ptrJPETSingleGammaModel ) { ptrJPETSingleGammaModel = new GateGammaSourceModelSingleGamma; }

 return ptrJPETSingleGammaModel;
}

