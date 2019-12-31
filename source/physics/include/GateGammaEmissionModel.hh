/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateGammaEmissionModel_hh
#define GateGammaEmissionModel_hh

#include "G4PrimaryParticle.hh"
#include <vector>
#include <string>
#include "GateEmittedGammaInformation.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Basic class for other model of gammas emission. Provides basic tools for calculation and can generate single gamma.
 **/
class GateGammaEmissionModel
{
 public:

  GateGammaEmissionModel();
  virtual ~GateGammaEmissionModel();
  
  /** Force gamma emissions in only one direction. When this method is called fUseFixedEmissionDirection flag is switched to TRUE.
   **/
  void SetFixedEmissionDirection( const G4ThreeVector& momentum_direction );
  /** Force gamma emissions in only one direction. When this method is  fUseFixedEmissionDirection flag is switched to value=enable.
   **/  
  void SetEnableFixedEmissionDirection( const G4bool enable );

  /** Set single gamma kinematic energy.
   **/
  void SetEmissionEnergy( const G4double& energy );
  /** Get single gamma kinematic energy.
   **/
  G4double GetEmissionEnergy() const;

  /** Set seed for generators from "Randomize.hh"
   **/
  void SetSeed( G4long seed );
  /** Get seed for generators from "Randomize.hh"
   **/
  G4long GetSeed() const;

  /** Generate single vertex with single gamma
   **/
  virtual G4int GeneratePrimaryVertices(G4Event* event, G4double& particle_time,  G4ThreeVector& particle_position);
   
 protected:
  /** Provides additional information for user about gamma
   **/
  virtual GateEmittedGammaInformation* GetPrimaryParticleInformation( const G4PrimaryParticle* pp, const GateEmittedGammaInformation::GammaKind& gamma_kind ) const;
  /** Generate random unit vector from uniform sphere distribution
   **/
  G4ThreeVector GetUniformOnSphere() const;
  /** Generate gamma polarization
   **/
  G4ThreeVector GetPolarization( const G4ThreeVector& momentum ) const;
  /** Generate perpendicular vector ( required for polarization )
   **/
  G4ThreeVector GetPerpendicularVector(const G4ThreeVector& v) const;
  /** Set model name ( class name )
   **/
  void SetModelName( const G4String model_name );
  /** Report error
   **/
  void NoticeError( G4String method_name, G4String exception_description ) const;
  /** Generate single gamma with desired energy
   **/
  G4PrimaryParticle* GetSingleGamma( const G4double& energy ) const;

 protected:
  G4ThreeVector fFixedEmissionDirection = G4ThreeVector( 0.0, 0.0, 0.0 );
  G4bool fUseFixedEmissionDirection = false;
  G4double fEmissionEnergy = 0.511 * MeV;//[MeV]
  G4String fModelName = "GateGammaEmissionModel";
  //Gamma definiotion - the same is used in each one model
  G4ParticleDefinition* pGammaDefinition = nullptr;

};

#endif
