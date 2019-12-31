/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateEmittedGammaInformation_hh
#define GateEmittedGammaInformation_hh

#include "G4VUserPrimaryParticleInformation.hh"
#include "G4ThreeVector.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: 
 **/
class GateEmittedGammaInformation : public G4VUserPrimaryParticleInformation
{
 public:
  GateEmittedGammaInformation();
  virtual ~GateEmittedGammaInformation();

  /** This enum specifies what is source of gamma.
      Each class which inherits from GateEmissionModel class should add
      here its own kinds.
  **/
  enum SourceKind
  {
   NotDefined = 0, // by default
   SingleGammaEmitter = 1, // just emitted single gamma with specfic energy ( by class GateGammaEmissionModel )
   ParaPositronium = 2, // 2 gammas ( plus prompt if it is required ) from pPs decay ( by GatePositroniumDecayModel )
   OrthoPositronium = 3 // 3 gammas ( plus prompt if it is required ) from oPs decay ( by GatePositroniumDecayModel )
  };

  /** This enum specifies model of source decay ( if it is present )
  **/
  enum DecayModel
  {
   None = 0, // Not specify - for example when there is not decay ( e.g. class GateGammaEmissionModel )
   Standard = 1, // Standard decay of source ( e.g. pPs -> 2 gamma )
   Deexcitation = 2 // If only deexcitation is present or when is part of standard decay changel ( Na22* -> Na22 + prompt gamma, Na22 -> Ne22 + e+, pPs-> 2 gamma <==> pPs* --> 2 gamma + prompt )
  };

  /** This enum specifies of gamma emitted by source
  **/
  enum GammaKind
  {
   Unknown = 0, // by default
   Single = 1, // gamma is emitted alone ( by class GateGammaEmissionModel )
   Annihilation = 2, // gamma is a product of annihilation
   Prompt = 3 // gamma is emitted from deexcitation process
  };

  void SetSourceKind( SourceKind source_kind );
  SourceKind GetSourceKind() const;

  void SetDecayModel( DecayModel decay_model );
  DecayModel GetDecayModel() const;

  void SetGammaKind( GammaKind gamma_kind );
  GammaKind GetGammaKind() const;

  /** Set polarization of gamma at the moment when it was emitted
   **/
  void SetInitialPolarization( const G4ThreeVector& polarization );
  /** Get polarization of gamma at the moment when it was emitted
   **/
  G4ThreeVector GetInitialPolarization() const;
  /** Set time shift - caused by positronium lifetime
   **/
  void SetTimeShift( const G4double& time_shift );
  /** Get time shift - caused by positronium lifetime
   **/  
  G4double GetTimeShift() const;

  virtual void Print() const;

 protected:
  SourceKind fSourceKind = SourceKind::NotDefined;
  DecayModel fDecayModel = DecayModel::None;
  GammaKind fGammaKind = GammaKind::Unknown;
  G4ThreeVector fInitialPolarization = G4ThreeVector( 0.0, 0.0, 0.0 );
  G4double fTimeShift = 0.0;//[ns]
};

#endif
