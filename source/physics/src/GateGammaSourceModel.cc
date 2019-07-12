/**
 *  @copyright Copyright 2017 The J-PET Gate Authors. All rights reserved.
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  @file GateGammaSourceModel.cc
 */

#include "GateGammaSourceModel.hh"
#include "TMath.h"

GateGammaSourceModel::GateGammaSourceModel() { ptrRandomGenerator = new TRandom3( 0 ); }

GateGammaSourceModel::~GateGammaSourceModel()
{
  delete ptrRandomGenerator;
}


void GateGammaSourceModel::SetPositronMomentum( const TVector3& boost ) { fPositronMomentum = boost; }


G4ThreeVector GateGammaSourceModel::GetPerpendicularVector( const G4ThreeVector& source_vector ) const
{
  G4double dx = source_vector.x();
  G4double dy = source_vector.y();
  G4double dz = source_vector.z();
  G4double x = dx < 0.0 ? -dx : dx;
  G4double y = dy < 0.0 ? -dy : dy;
  G4double z = dz < 0.0 ? -dz : dz;
  if (x < y)
    return x < z ? G4ThreeVector(-dy,dx,0) : G4ThreeVector(0,-dz,dy);
  else
    return y < z ? G4ThreeVector(dz,0,-dx) : G4ThreeVector(-dy,dx,0);
}

G4ThreeVector GateGammaSourceModel::GetPolarization( const G4ThreeVector& momentum_direction ) const
{
 /***
  * Gamma polarization and direction are connected, and we can't set any value to polarization.
  * Compton's scattering algorithm (e.g. G4LivermorePolarizationComptonModel.cc) prohibit polarization which is not orthogonal to gamma direction.
  * Depends on how orthogonal is not this polarization Compton's algorithm generate two types of polarization: (p0 - prime polarization, d0 - gamma direction, p - new polarization)
  * 1. If p0 is not orthogonal to d0 or p0 is zero vector (p0={0,0,0}) algorithm generate random orthogonal to d0 polarization p. Sometimes is good - e.g. we want obtain uniform phi angle
  * distribution in Klein–Nishina cross-section plot  - in this situation always use zero vector in other case is always error and you will receive phi angle distribution in situation where it shoudn't be.
  * 2. p0 is not exactly orthogonal to d0 - what mean dot product of d0 and p0 is not equal zero - in this situation algorithm  extract orthogonal part from p0 : p = p0 - p0.dot(d0)/d0.dot(d0) * d0;
  * All above is only information for future users  - in this code I provide orthogonality of polarization as orthogonal product from gamma direction.
  *
  * More important information about Klein–Nishina algorithm here : polarization is threat as 3D vector in the Cartesian system - this isn't Stocke's vector.
  * User can define only linear polarization by angle or lack of polarization.
  * For define linear polarization use command : PolarizationAngleEnable
  * For use unpolarized gamma use command : UnpolarizedGammaEnable
  *
  * Linear polarization is defined as a vector on plane orthogonal to gamma direction.
  * */

 G4ThreeVector polarization = {0,0,0};
 G4ThreeVector d0 = momentum_direction.unit();
 if(!fUseUnpolarizedGamma)
 {
  if ( fUseRandomAnglesForPolarizations )
   polarization = calcPolarization( d0, ptrRandomGenerator->Uniform( 0.0, M_PI ) );
  else
   polarization = calcPolarization( d0, fPolarizationAngleInRadians );
 }
 return polarization.unit();
}

G4ThreeVector GateGammaSourceModel::calcPolarization( G4ThreeVector& d0, double angle_radians ) const
{
 G4ThreeVector a0,b0;
 a0 = GetPerpendicularVector( d0 ).unit();
 b0 = d0.cross( a0 ).unit();
 G4ThreeVector polarization = std::cos( angle_radians ) * a0 + std::sin( angle_radians ) * b0;
 polarization.unit();
 polarization -= polarization.dot( d0 ) * d0;
 return polarization;
}

G4ThreeVector GateGammaSourceModel::GetPerpendicularPolarizationToItsMomentumAndOtherPolarization( const G4ThreeVector& own_momentum_direction, const G4ThreeVector& other_polarization ) const
{
 return own_momentum_direction.unit().cross( other_polarization.unit() ).unit();
}

void GateGammaSourceModel::SetLinearPolarizationAngle(double angle, bool is_degree)
{
 fPolarizationAngleInRadians = is_degree ? angle * TMath::DegToRad() : angle;
 fUseRandomAnglesForPolarizations = false;
}

void GateGammaSourceModel::SetParticlesNumber( int particles_number ) { fParticlesNumber = particles_number; }

int GateGammaSourceModel::GetParticlesNumber() const { return fParticlesNumber; }

void GateGammaSourceModel::SetUnpolarizedGammaGeneration( bool use_unpolarized ) { fUseUnpolarizedGamma = use_unpolarized; }

bool GateGammaSourceModel::GetIsUnpolarizedGammaGenerationInUse() const { return fUseUnpolarizedGamma; }

TVector3 GateGammaSourceModel::GetPositronMomentum() const { return fPositronMomentum; }

void GateGammaSourceModel::SetSeedForRandomGenerator( unsigned int seed )
{
 fSeedForRandomGenerator = seed;

 GetRandomGenerator()->SetSeed( fSeedForRandomGenerator );
}

unsigned int GateGammaSourceModel::GetSeedForRandomGenerator() const { return fSeedForRandomGenerator; }

void GateGammaSourceModel::SetPromptGammaEnergy( double energy ) { fPromptGammaEnergy = energy; }

double GateGammaSourceModel::GetPromptGammaEnergy() const { return fPromptGammaEnergy; }

TRandom3* GateGammaSourceModel::GetRandomGenerator()
{ 
 if ( ptrRandomGenerator == nullptr ) { ptrRandomGenerator = new TRandom3( fSeedForRandomGenerator ); }

 return ptrRandomGenerator; 
}

GateGammaModelPrimaryParticleInformation* GateGammaSourceModel::GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind kind, const G4ThreeVector& polarization ) const
{
 GateGammaModelPrimaryParticleInformation* info = new GateGammaModelPrimaryParticleInformation();
 info->setGammaSourceModel( fGammaSourceModel );
 info->setGammaKind( kind );
 info->setInitialPolarization( polarization );
 return info;
}

void GateGammaSourceModel::SetGammaSourceModel( GateGammaModelPrimaryParticleInformation::GammaSourceModel model ) { fGammaSourceModel = model; }
