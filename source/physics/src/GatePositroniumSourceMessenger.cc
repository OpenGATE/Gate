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
 upCmdSetPositroniumLifetimes.reset(GetStringCmd( "setPositroniumLifetimes", "\"t1, t2, t3 .., tn\" - where ti corresponds to lifetime constants of the components" ) );
 upCmdSetDecayKinds.reset(GetStringCmd( "setDecayKinds", "\"k1, k2, k3 .., kn\" - where ki is k2Gamma or k3Gamma" ) );
 upCmdSetPositronInteractions.reset(GetStringCmd( "setPositronInteractions", "\"k1, k2, k3 .., kn\" - where ki is kParaPs, kDirect or kOrthoPs, of a given element in vector of components. Used to properly recalculate intensities of the components from the theory" ) );
 upCmdSetPromptPhotonProbabilites.reset(GetStringCmd( "setPromptPhotonProbabilites", "\"f1, f2, f3 .., fn\" - where fi in [0.0, 1.0] " ) );
 upCmdSetPromptPhotonEnergies.reset(GetStringCmd( "setPromptPhotonEnergies", "\"e1, e2, e3 .., fn\" - where ei are energies " ) );
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
    std::vector<float> lifetimes;
    std::stringstream ss(new_value);
    std::string num;
    while (ss >> num) {
      if (ss.good()) {
        lifetimes.push_back(stod(num));
      } else {
        G4double unitVal = CheckIfUnit(num);
      }
    }
    for (unsigned i=0; i<lifetimes.size(); i++)
      lifetimes.at(i) *= unitVal;
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
    std::vector<float> promptPhotonEnergies;
    std::stringstream ss(new_value);
    std::string energy;
    while (ss >> energy) {
      if (ss.good()) {
        promptPhotonEnergies.push_back(stod(energy));
      } else {
        G4double unitVal = CheckIfUnit(energy);
      }
    }
    for (unsigned i=0; i<promptPhotonEnergies.size(); i++)
      promptPhotonEnergies.at(i) *= unitVal;
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

G4double GatePositroniumSourceMessenger::CheckIfUnit(std::string val)
{
  int strSize = val.size();
  G4double unitValue = 1;

  if (!strSize || strSize > 3)
    return unitValue;

  char firstChar = val.at(0);
  char lastChar = val.at(strSize-1);
  if (lastChar == 's') {
    unitValue = 1.e+9;          // default is ns
  } else if (lastChar == 'm') {
    unitValue = 1.e+4;          // default is mm
  } else if (lastChar == 'V' && strSize > 1) {
    if (val.at(strSize-2) == 'e')
      unitValue = 1.e-6;        // default is MeV
  } // Bq is the nominal value therefore does not need additional handling

  if (firstChar == 'G') {
    unitValue *= 1.e+9;
  } else if (firstChar == 'M') {
    unitValue *= 1.e+6;
  } else if (firstChar == 'k') {
    unitValue *= 1.e+3;
  } else if (firstChar == 'c') {
    unitValue *= 1.e-2;
  } else if (firstChar == 'm') {
    unitValue *= 1.e-3;
  } else if (firstChar == 'u') {
    unitValue *= 1.e-6;
  } else if (firstChar == 'n') {
    unitValue *= 1.e-9;
  }

  return unitValue;
}
