/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef PositroniumDecayModel_hh
#define PositroniumDecayModel_hh

#include<vector>
#include "G4DecayTable.hh"
#include "G4ParticleTable.hh"
#include "G4PrimaryParticle.hh"
#include "G4GeneralPhaseSpaceDecay.hh"
#include "G4DecayTable.hh"
#include "G4ParticleDefinition.hh"
#include "GateEmittedGammaInformation.hh"
#include "GateGammaEmissionModel.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "G4VDecayChannel.hh"
#include "G4ParticleDefinition.hh"
#include "G4PrimaryVertex.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Class generate gammas from positronium decay with including deexcitation gamma ( prompt gamma ) if is required.
 **/
class GatePositroniumDecayModel : public GateGammaEmissionModel
{
 public:

  /**  About class: representation of positronium for main class. Provides access to positronium decay channel.
   **/
  class Positronium
  {
   public:
    Positronium( G4String name, G4double life_time, G4int annihilation_gammas_number );
    void SetLifeTime( const G4double& life_time );
    G4double GetLifeTime() const;
    G4String GetName() const;
    G4int GetAnnihilationGammasNumber() const;
    G4DecayProducts* GetDecayProducts();
   private:
    G4String fName = "";
    G4double fLifeTime = 0.0;//[ns]
    G4int fAnnihilationGammasNumber = 0;
    G4VDecayChannel* pDecayChannel = nullptr;
  };

  /** Positronium kind tells which positronium we will use.
    * Depends on used Ps gammas number and time will be different.
  **/
  enum PositroniumKind { pPs, oPs };
  /** Decay model descibes decay of positronium.
    * For example for PositroniumKind::pPs we have decays:
    * 1) Standard: pPs -> 2 gamma
    * 2) WithPrompt: pPs* -> 2 gamma + prompt_gamma
    * Only prompt gamma has nod modified time, other gammas always has modified time.
  **/
  enum DecayModel { Standard, WithPrompt };

  GatePositroniumDecayModel();
  virtual ~GatePositroniumDecayModel();

  void SetPositroniumKind( PositroniumKind positronium_kind );
  PositroniumKind GetPositroniumKind() const;

  void SetDecayModel( DecayModel decay_model );
  DecayModel GetDecayModel() const;

  void SetPostroniumLifetime( G4String positronium_name, G4double life_time ); //[ns]

  void SetPromptGammaEnergy( G4double prompt_energy ); //[keV]
  G4double GetPromptGammaEnergy() const; //[keV]

  /** Set probability of gammas emission from para-positronium decay ( and ortho-positronium too )
   * @param: fraction - number in range from 0.0 to 1.0 ( 0.0 - generate only gammas from oPs decay, 1.0 - generate only gammas from pPs decay )
   **/
  void SetParaPositroniumFraction( G4double fraction );

  /** Generate vertices and fill them with gammas. Vertices number depends on used DecayModel:
    *  - one ( if DecayModel::Standard )
    *  - two ( if DecayModel::WithPrompt )
   **/
  virtual G4int GeneratePrimaryVertices(G4Event* event, G4double& particle_time,  G4ThreeVector& particle_position) override;

 protected:
  /** Provides additional information for user about gamma
   **/
  virtual GateEmittedGammaInformation* GetPrimaryParticleInformation( const G4PrimaryParticle* pp, const GateEmittedGammaInformation::GammaKind& gamma_kind ) const override;
  /** Depends on used model and setted fractions it chooses positronium which decay will be used to generate gammas
   **/
  void PreparePositroniumParametrization();
  /** Generate vertex for deexcitation gamma ( prompt gamma ) - position and time is the same as generted by source 
   **/
  G4PrimaryVertex* GetPrimaryVertexFromDeexcitation(const G4double& particle_time, const  G4ThreeVector& particle_position );
  /** Generate vertex for annihilation gammas  - position is the same as generted by source, but time is shifted by positronium lifetime ( T0 + f(lifetime))
   **/
  G4PrimaryVertex* GetPrimaryVertexFromPositroniumAnnihilation( const G4double& particle_time, const  G4ThreeVector& particle_position );
  /** Generate deexcitation ( prompt ) gamma
   **/
  G4PrimaryParticle* GetGammaFromDeexcitation();
  /** Generate annihilation gammas
   **/
  std::vector<G4PrimaryParticle*> GetGammasFromPositroniumAnnihilation();

 protected:
  //Positronium model - for para-positronium
  Positronium fParaPs = Positronium( "pPs", 0.1244 * ns, 2 );
  //Positronium model - for ortho-positronium
  Positronium fOrthoPs = Positronium( "oPs", 138.6 * ns, 3 );
  //Positronium model - for current event
  Positronium* pInfoPs = nullptr;
  //Default deexcitation gamma energy - if user didn't set prompt gamma energy this value will be used
  const G4double kSodium22DeexcitationGammaEnergy = 1.022 * MeV;//[MeV]
  
  //Which positronium use for gammas generation
  PositroniumKind fPositroniumKind = PositroniumKind::pPs;
  //Which decay model use for gammas generation
  DecayModel fDecayModel = DecayModel::Standard;
  //Dexcitation gamma energy
  G4double fPromptGammaEnergy = kSodium22DeexcitationGammaEnergy;//[MeV]
  //Propability of emiiting gammas from para-positronium ( number in range from 0.0 to 1.0 )
  G4double fParaPositroniumFraction = 1.0;
  //It is required to generate mixed positronium decays ( pPs and oPs witch propability controled by varaible fParaPositroniumFraction )
  G4bool fUsePositroniumFractions = false;
  
};

#endif
