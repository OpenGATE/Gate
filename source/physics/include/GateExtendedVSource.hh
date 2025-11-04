/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateExtendedVSource_hh
#define GateExtendedVSource_hh

#include <optional>
#include <memory>
#include "GateVSource.hh"
#include "GateExtendedVSourceMessenger.hh"
#include "GateGammaEmissionModel.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Refactored by: Wojciech Krzemien
 *  About class: Extended version of GateVSource. It focuses on generating gammas from positronium decay.
 **/
class GateExtendedVSource : public GateVSource
{
public:

  enum class ModelKind { 
   NotDefined, //by default - in this case this class will behave like GateVSource
   SingleGamma, // generate single gamma
   ParaPositronium, //generate gammas from para-positronium decay
   OrthoPositronium, //generate gammas from ortho-positronium decay
   Positronium, //generate gammas from mixed model ( from pPs and oPs decay with setted ratio )
   MiniPositronium //generate gammas from extended mixed model 
  };

  explicit GateExtendedVSource(const G4String& name);
  virtual ~GateExtendedVSource() = default;

  /** Generate gammas for event
   **/
  virtual G4int GeneratePrimaries( G4Event* event ) override;

  /** Set fixed direction of single gamma ( or prompt gamma )
   **/
  void SetFixedEmissionDirection(const G4ThreeVector &fixed_emission_direction);
  /** Set enable/disable emission of single gamma with fixed direction
   **/
  void SetEnableFixedEmissionDirection(G4bool enable_fixed_emission_direction);
  /** Set single gamma kinetic energy
   **/
  void SetEmissionEnergy(G4double energy);
  void SetSeed(G4long seed);

 protected:
  /** Set model used for this source. If is not defined then this class will behave like GateVSource.
   **/
   void SetModel(const G4String &model_name);
   /** Prepare model to work - set all settings from user to model
    **/
   void PrepareModel();

 protected:
  //Gamma emission model
   std::unique_ptr<GateGammaEmissionModel> pModel;
  //Source messanger
   std::unique_ptr<GateExtendedVSourceMessenger> pMessenger;
  //User settings:
  ModelKind fModelKind = ModelKind::NotDefined;
  std::optional<G4ThreeVector> fFixedEmissionDirection;
  std::optional<G4bool> fEnableFixedEmissionDirection;
  std::optional<G4double> fEmissionEnergy;
  std::optional<G4long> fSeed;
  
  //Set by PrepareModel() and used in GeneratePrimaries()
  G4bool fBehaveLikeVSource = false;

  //Constants for Set(...) methods
  static inline const G4String kParaPositroniumName = "pPs";
  static inline const G4String kOrthoPositroniumName = "oPs";
};

#endif
