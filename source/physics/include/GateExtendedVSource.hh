/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateExtendedVSource_hh
#define GateExtendedVSource_hh

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
   ParaPositronium, //generate gammas from para-positronium decay
   OrthoPositronium, //generate gammas from ortho-positronium decay
   Positronium}; // generate gammas from mixed model

  explicit GateExtendedVSource(const G4String& name);
  virtual ~GateExtendedVSource() = default;

  /** Generate gammas for event
   **/
  virtual G4int GeneratePrimaries( G4Event* event ) override;

 protected:
  /** Set model used for this source. If is not defined then this class will behave like GateVSource.
   **/
   void SetModel(const G4String &model_name);
   /** Prepare model to work - set all settings from user to model
    **/
   void PrepareModel();

 protected:
   std::unique_ptr<GateGammaEmissionModel> pModel;
   std::unique_ptr<GateExtendedVSourceMessenger> pMessenger;
  //User settings:
  ModelKind fModelKind = ModelKind::NotDefined;
  
  //Set by PrepareModel() and used in GeneratePrimaries()
  G4bool fBehaveLikeVSource = false;
};

#endif
