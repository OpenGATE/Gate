/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateExtendedVSource_hh
#define GateExtendedVSource_hh

#include "GateVSource.hh"
#include "GateExtendedVSourceMessenger.hh"
#include "GateGammaEmissionModel.hh"

/**  About class: this is helper class to control if setting is in use
 **/
template <class T>
class ModelSetting 
{
 public:
  void Set( T value )
  { 
   fValue = value;
   fIsSetted = true;
  }
  T Get() const { return fValue; }
  bool IsSetted() const { return fIsSetted; }
 private:
  T fValue;
  bool fIsSetted = false;
};

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Extended version of GateVSource. It focuses on generating gammas from positronium decay.
 **/
class GateExtendedVSource : public GateVSource
{
public:

  enum class ModelKind { 
   NotDefined, //by default - in this case this clas will behave like GateVSource
   SingleGamma, // generate single gamma
   ParaPositronium, //generate gammas from para-positronium decay
   OrthoPositronium, //generate gammas from ortho-positronium decay
   Positronium //generate gammas from mixed model ( from pPs and oPs decay with setted ratio )
  };

  GateExtendedVSource( G4String name );
  virtual ~GateExtendedVSource();

  /** Generate gammas for event
   **/
  virtual G4int GeneratePrimaries( G4Event* event ) override;

  /** Set enable emission of additional gamma - from deexcitation ( prompt gamma )
   **/
  void SetEnableDeexcitation( const G4bool& enable_deexcitation );
  /** Set fixed direction of single gamma ( or prompt gamma )
   **/
  void SetFixedEmissionDirection( const G4ThreeVector& fixed_emission_direction );
  /** Set enable/disable emission of single gamma with fixed direction
   **/
  void SetEnableFixedEmissionDirection( const G4bool& enable_fixed_emission_direction );
  /** Set single gamma kinetic energy
   **/
  void SetEmissionEnergy( const G4double& energy );
  /** Set seed for Randomize.hh generatores
   **/
  void SetSeed( const G4long& seed );
  /** Set positronium lifetime - which is included as constant for exponential distribution ( G4RandExponential )
    * @param: positronium_name - for example: pPs, oPs
    * @param: life_time - in ns
    * Lifetime value will have inpact of vertex time set for annihilation gammas
   **/
  void SetPostroniumLifetime( const G4String& positronium_name ,const G4double& life_time );
  /** Set prompt gamma energy ( deexcictation energy ). 
    * If user set enable emission of prompt gamma without set prompt energy then il wii be default value used ( deexcitation of Na22 )
   **/
  void SetPromptGammaEnergy( const G4double& energy );
  /** Set propability of gammas emission from different positronium.
    * @param: positronium_kind - positronium name: pPs, oPs
    * @param: fraction - number in range from 0.0 to 1.0
    * You have to call this method for only one kind of positronium - for the second one propability will be calculated as: 1.0 - fraction.
  **/
  void SetPositroniumFraction( const G4String& positronium_kind, const G4double& fraction );

 protected:
  /** Set model used for this source. If is not defined then this class will behave like GateVSource.
   **/
  void SetModel( const G4String& model_name );
  /** Prepare model to work - set all settings from user to model
   **/
  void PrepareModel();

 protected:
  //Gamma emission model
  GateGammaEmissionModel* pModel = nullptr;
  //Source messanger
  GateExtendedVSourceMessenger* pMessenger = nullptr;
  //User settings:
  ModelKind fModelKind = ModelKind::NotDefined;
  ModelSetting<G4bool> fEnableDeexcitation;
  ModelSetting<G4ThreeVector> fFixedEmissionDirection;
  ModelSetting<G4bool> fEnableFixedEmissionDirection;
  ModelSetting<G4double> fEmissionEnergy;
  ModelSetting<G4long> fSeed;
  ModelSetting<G4double> fParaPostroniumLifetime;
  ModelSetting<G4double> fOrthoPostroniumLifetime;
  ModelSetting<G4double> fPromptGammaEnergy;
  ModelSetting<G4double> fParaPositroniumFraction;
  //Constants for Set(...) methods
  const G4String kParaPositroniumName = "pPs";
  const G4String kOrthoPositroniumName = "oPs";
  //Set by PrepareModel() and used in GeneratePrimaries()
  G4bool fBehaveLikeVSource = false;

};

#endif
