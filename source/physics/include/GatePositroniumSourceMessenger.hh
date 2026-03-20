/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GatePositroniumSourceMessenger_hh
#define GatePositroniumSourceMessenger_hh

#include <memory>

#include "GateVSourceMessenger.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GatePositroniumDecayParamsGenerator.hh"

class GatePositroniumSource;

/** 
 *  About class: Messenger for GatePositroniumSource class
 **/
class GatePositroniumSourceMessenger: public GateVSourceMessenger
{
 public:
  explicit GatePositroniumSourceMessenger(GatePositroniumSource *source);
  ~GatePositroniumSourceMessenger()=default;

  void SetNewValue(G4UIcommand *command, G4String newValue) override;
   
  PositroniumDecayModelParams generatePositroniumDecayParams(GatePositroniumDecayParamsGenerator::DecayModel model= GatePositroniumDecayParamsGenerator::kPositronium) const;

 protected:
  void InitCommands();

  G4UIcmdWithABool* GetBoolCmd(const G4String& cmd_name, const G4String& cmd_guidance);
  G4UIcmdWithADoubleAndUnit* GetDoubleCmdWithUnit(const G4String& cmd_name, const G4String&  cmd_guidance, const G4String&  default_unit, const G4String&  unit_candidates);
  G4UIcmdWith3Vector* GetVectorCmd(const G4String& cmd_name, const G4String&  cmd_guidance);
  G4UIcmdWithAnInteger* GetIntCmd(const G4String& cmd_name, const G4String&  cmd_guidance);
  G4UIcmdWithAString* GetStringCmd(const G4String& cmd_name, const G4String&  cmd_guidance);
  G4UIcmdWith3VectorAndUnit* GetVectorCmdWithUnit(const G4String& cmd_name, const G4String& cmd_guidance, const G4String& default_unit, const G4String& unit_candidates);

 protected:
  GatePositroniumSource* pSource = nullptr;

  std::unique_ptr<G4UIcmdWithAString> upCmdSetPositroniumFractions;
  std::unique_ptr<G4UIcmdWithAString> upCmdSetPositroniumLifetimes;
  std::unique_ptr<G4UIcmdWithAString> upCmdSetPromptPhotonProbabilites;
  std::unique_ptr<G4UIcmdWithAString> upCmdSetPromptPhotonEnergies;
  std::unique_ptr<G4UIcmdWithAString> upCmdSetDecayKinds;
  std::unique_ptr<G4UIcmdWithAString> upCmdSetPositronInteractions;

  std::unique_ptr<G4UIcmdWithAString> upCmdSetMeanPositronRange;

  std::vector<float> parseListOfParamsWithUnit(const G4String& input) const;

  GatePositroniumDecayParamsGenerator fParamGenerator;
  
};

#endif
