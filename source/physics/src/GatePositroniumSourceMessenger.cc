/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include <sstream>

#include "GatePositroniumSourceMessenger.hh"
#include "GatePositroniumSource.hh"

GatePositroniumSourceMessenger::GatePositroniumSourceMessenger(GatePositroniumSource *source): GateVSourceMessenger(source), pSource(source) 
{
  InitCommands();
}

G4UIcmdWithABool* GatePositroniumSourceMessenger::GetBoolCmd(const G4String& cmd_name, const G4String& cmd_guidance )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWithABool* cmd = new G4UIcmdWithABool( cmd_path, this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( cmd_name, false );
 return cmd;
}

G4UIcmdWithADoubleAndUnit* GatePositroniumSourceMessenger::GetDoubleCmdWithUnit( const G4String& cmd_name, const G4String& cmd_guidance, const G4String& default_unit, const G4String& unit_candidates )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWithADoubleAndUnit* cmd = new G4UIcmdWithADoubleAndUnit( cmd_path , this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( cmd_name, false );
 cmd->SetDefaultUnit( default_unit.c_str() );
 cmd->SetUnitCandidates( unit_candidates.c_str() );
 return cmd;
}

G4UIcmdWith3Vector* GatePositroniumSourceMessenger::GetVectorCmd( const G4String& cmd_name, const G4String& cmd_guidance )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWith3Vector* cmd = new G4UIcmdWith3Vector( cmd_path, this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( G4String( cmd_name + "_x" ), G4String( cmd_name + "_y" ), G4String( cmd_name + "_z" ), false );
 return cmd;
}

G4UIcmdWithAnInteger* GatePositroniumSourceMessenger::GetIntCmd( const G4String& cmd_name, const G4String& cmd_guidance )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWithAnInteger* cmd = new G4UIcmdWithAnInteger( cmd_path, this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( cmd_name, false );
 return cmd;
}

G4UIcmdWithAString* GatePositroniumSourceMessenger::GetStringCmd(const G4String& cmd_name, const G4String& cmd_guidance )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWithAString* cmd = new G4UIcmdWithAString( cmd_path, this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( cmd_name, false );
 return cmd;
}

G4UIcmdWith3VectorAndUnit* GatePositroniumSourceMessenger::GetVectorCmdWithUnit( const G4String& cmd_name, const G4String& cmd_guidance, const G4String& default_unit, const G4String& unit_candidates )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWith3VectorAndUnit* cmd = new G4UIcmdWith3VectorAndUnit( cmd_path , this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( G4String( cmd_name + "_x" ), G4String( cmd_name + "_y" ), G4String( cmd_name + "_z" ), false );
 cmd->SetDefaultUnit( default_unit.c_str() );
 cmd->SetUnitCandidates( unit_candidates.c_str() );
 return cmd; 
}

void GatePositroniumSourceMessenger::InitCommands()
{
 upCmdSetPositroniumFractions.reset(GetStringCmd( "setPositroniumFractions", "\"f1, f2, f3 .., fn\" - where fi in [0.0, 1.0] and sum of all fi ==1" ) );
 upCmdSetPositroniumLifetimes.reset(GetStringCmd( "setPositroniumLifetimes", "\"t1, t2, t3 .., tn t_unit\" - where ti corresponds to lifetime constants and t_unit is one of the Geant4 time units e.g. ns" ) );
 upCmdSetDecayKinds.reset(GetStringCmd( "setDecayKinds", "\"k1, k2, k3 .., kn\" - where ki is k2Gamma or k3Gamma" ) );
 upCmdSetPositronInteractions.reset(GetStringCmd( "setPositronInteractions", "\"k1, k2, k3 .., kn\" - where ki is kParaPs, kDirect or kOrthoPs, of a given element in vector of components. Used to properly recalculate intensities of the components from the theory" ) );
 upCmdSetPromptPhotonProbabilites.reset(GetStringCmd( "setPromptPhotonProbabilites", "\"f1, f2, f3 .., fn\" - where fi in [0.0, 1.0] " ) );
 upCmdSetPromptPhotonEnergies.reset(GetStringCmd( "setPromptPhotonEnergies", "\"e1, e2, e3 .., en e_unit\" - where ei are energies and e_unit is one of the Geant4 energy units e.g. MeV" ) );
}

std::vector<float> GatePositroniumSourceMessenger::parseListOfParamsWithUnit(const G4String& input) const
{
  std::vector<float> values;
  std::stringstream ss(input);
  std::vector<G4String> tokens;
  std::string token;

  while (ss >> token) {
    tokens.push_back(token);
  }

  if (tokens.size() < 2) {
    GateError("parseListOfParamsWithUnit(): Need at least one value and one unit.");
  }

  G4String unitStr = tokens.back();
  G4double unitValue = G4UnitDefinition::GetValueOf(unitStr);

  for (unsigned i=0; i<tokens.size()-1; i++) {
    values.push_back(std::stod(tokens[i]) * unitValue);
  }

  return values;
}  

void GatePositroniumSourceMessenger::SetNewValue(G4UIcommand *command, G4String new_value) 
{
  if (command == upCmdSetPositroniumFractions.get()) {
    std::vector<float> fractions;
    std::stringstream ss(new_value);
    G4double num;
    while (ss >> num) {
      fractions.push_back(num);
    }
    fParamGenerator.SetPositroniumFraction(fractions);
  } else if (command == upCmdSetPositroniumLifetimes.get()) {
    auto lifetimes =parseListOfParamsWithUnit(new_value);
    fParamGenerator.SetPositroniumLifetimes(lifetimes);
  } else if (command == upCmdSetPromptPhotonProbabilites.get()) {
    std::vector<float> promptPhotonProb;
    std::stringstream ss(new_value);
    float prob;
    while (ss >> prob) {
      prob = prob < 0 ? 0 : prob;
      prob = prob > 1 ? 1 : prob;
      promptPhotonProb.push_back(prob);
    }
    fParamGenerator.SetPromptGammaProbabilities(promptPhotonProb);
  } else if (command == upCmdSetPromptPhotonEnergies.get()) {
    auto promptPhotonEnergies = parseListOfParamsWithUnit(new_value);
    fParamGenerator.SetPromptGammaEnergies(promptPhotonEnergies);
  } else if (command == upCmdSetDecayKinds.get()) {
    std::vector<PositroniumDecayKind> decayKinds;
    std::stringstream ss(new_value);
    std::string kind;
    while (ss >> kind) {
      if (kind == "k2Gamma") {
        decayKinds.push_back(k2Gamma);
      } else {
        if (kind == "k3Gamma") {
          decayKinds.push_back(k3Gamma);
        } else {
          GateError("GatePositroniumSourceMessenger::SetNewValue: unknown "
                    "decay kind read from macro.");
        }
      }
    }
    fParamGenerator.SetDecayKinds(decayKinds);
  } else if (command == upCmdSetPositronInteractions.get()) {
    std::vector<PositronElectronInteraction> positronInteractions;
    std::stringstream ss(new_value);
    std::string inter;
    while (ss >> inter) {
      if (inter == "kParaPs") {
        positronInteractions.push_back(PositronElectronInteraction::kParaPs);
      } else if (inter == "kDirect") {
        positronInteractions.push_back(PositronElectronInteraction::kDirect);
      } else if (inter == "kOrthoPs") {
        positronInteractions.push_back(PositronElectronInteraction::kOrthoPs);
      } else {

        GateError("GatePositroniumSourceMessenger::SetNewValue: unknown "
                  "interaction type read from macro.");
      }
    }
    fParamGenerator.SetPositronInteractions(positronInteractions);
  } else {
    GateVSourceMessenger::SetNewValue(command, new_value);
  }
}

PositroniumDecayModelParams GatePositroniumSourceMessenger::generatePositroniumDecayParams(GatePositroniumDecayParamsGenerator::DecayModel model) const
{
  return fParamGenerator.generatePositroniumDecayParams(model);
}
