/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include <sstream>

#include "GateExtendedVSourceMessenger.hh"
#include "GateExtendedVSource.hh"

GateExtendedVSourceMessenger::GateExtendedVSourceMessenger(GateExtendedVSource *source): GateVSourceMessenger(source), pSource(source) 
{
  InitCommands();
}

G4UIcmdWithABool* GateExtendedVSourceMessenger::GetBoolCmd(const G4String& cmd_name, const G4String& cmd_guidance )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWithABool* cmd = new G4UIcmdWithABool( cmd_path, this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( cmd_name, false );
 return cmd;
}

G4UIcmdWithADoubleAndUnit* GateExtendedVSourceMessenger::GetDoubleCmdWithUnit( const G4String& cmd_name, const G4String& cmd_guidance, const G4String& default_unit, const G4String& unit_candidates )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWithADoubleAndUnit* cmd = new G4UIcmdWithADoubleAndUnit( cmd_path , this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( cmd_name, false );
 cmd->SetDefaultUnit( default_unit.c_str() );
 cmd->SetUnitCandidates( unit_candidates.c_str() );
 return cmd;
}

G4UIcmdWith3Vector* GateExtendedVSourceMessenger::GetVectorCmd( const G4String& cmd_name, const G4String& cmd_guidance )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWith3Vector* cmd = new G4UIcmdWith3Vector( cmd_path, this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( G4String( cmd_name + "_x" ), G4String( cmd_name + "y" ), G4String( cmd_name + "z" ), false );
 return cmd;
}

G4UIcmdWithAnInteger* GateExtendedVSourceMessenger::GetIntCmd( const G4String& cmd_name, const G4String& cmd_guidance )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWithAnInteger* cmd = new G4UIcmdWithAnInteger( cmd_path, this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( cmd_name, false );
 return cmd;
}

G4UIcmdWithAString* GateExtendedVSourceMessenger::GetStringCmd(const G4String& cmd_name, const G4String& cmd_guidance )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWithAString* cmd = new G4UIcmdWithAString( cmd_path, this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( cmd_name, false );
 return cmd;
}

G4UIcmdWith3VectorAndUnit* GateExtendedVSourceMessenger::GetVectorCmdWithUnit( const G4String& cmd_name, const G4String& cmd_guidance, const G4String& default_unit, const G4String& unit_candidates )
{
 G4String cmd_path = GetDirectoryName() + cmd_name;
 G4UIcmdWith3VectorAndUnit* cmd = new G4UIcmdWith3VectorAndUnit( cmd_path , this );
 cmd->SetGuidance( cmd_guidance );
 cmd->SetParameterName( G4String( cmd_name + "_x" ), G4String( cmd_name + "y" ), G4String( cmd_name + "z" ), false );
 cmd->SetDefaultUnit( default_unit.c_str() );
 cmd->SetUnitCandidates( unit_candidates.c_str() );
 return cmd; 
}

void GateExtendedVSourceMessenger::InitCommands()
{
 upCmdSetEnableDeexcitation.reset( GetBoolCmd( "setEnableDeexcitation", "Generate prompt gamma from positron source which precedes positronium formation and decay." ) );
 upCmdSetFixedEmissionDirection.reset( GetVectorCmd( "setFixedEmissionDirection", "Set fixed direction for single and prompt gamma." ) );
 upCmdSetEnableFixedEmissionDirection.reset( GetBoolCmd( "setEnableFixedEmissionDirection", "Set fixed direction enable/disable." ) );
 upCmdSetEmissionEnergy.reset( GetDoubleCmdWithUnit( "setEmissionEnergy", "Set energy for single gamma.", "keV", "keV MeV GeV" ) );
 upCmdSetSeed.reset( GetIntCmd("setSeed", "Set seed for random generator" ) );
 upCmdSetPositroniumLifetime.reset( GetStringCmd( "setPositroniumLifetime", "Set life-time of positronium." ) ); 
 upCmdSetLifetime.reset( GetDoubleCmdWithUnit( "setLifetime", "Set life-time of positronium - disable for user.", "ns", "ps ns" ) ); 
 upCmdSetPromptGammaEnergy.reset( GetDoubleCmdWithUnit( "setPromptGammaEnergy", "Set energy for prompt gamma.", "keV", "keV MeV GeV" ) );
 upCmdSetPositroniumFraction.reset( GetStringCmd( "setPositroniumFraction", "\"positronium_kind fraction\" - where positronium_kind = {pPs, oPs} and fraction in [0.0, 1.0]" ) );


 upCmdSetPositroniumFractions.reset(GetStringCmd( "setPositroniumFractions", "\"f1, f2, f3 .., fn\" - where fi in [0.0, 1.0] and sum of all fi ==1" ) );
 upCmdSetPositroniumLifetimes.reset(GetStringCmd( "setPositroniumLifetimes", "\"t1, t2, t3 .., tn\" - where" ) );
 upCmdSetDecayKinds.reset(GetStringCmd( "setDecayKinds", "\"k1, k2, k3 .., kn\" - where ki " ) );
 upCmdSetIsPromptPhoton.reset(GetStringCmd( "setIsPromptPhoton", "\"f1, f2, f3 .., fn\" - where fi are true or false " ) );
 upCmdSetPromptPhotonEnergies.reset(GetStringCmd( "setPromptPhotonEnergies", "\"e1, e2, e3 .., fn\" - where ei are energies " ) );
}

void GateExtendedVSourceMessenger::SetNewValue( G4UIcommand* command, G4String new_value )
{
 if ( command == upCmdSetEnableDeexcitation.get() )
 {
  pSource->SetEnableDeexcitation( upCmdSetEnableDeexcitation->GetNewBoolValue( new_value ) );
 }
 else if ( command == upCmdSetFixedEmissionDirection.get() )
 {
  pSource->SetFixedEmissionDirection( upCmdSetFixedEmissionDirection->GetNew3VectorValue( new_value ) );
 }
 else if ( command == upCmdSetEnableFixedEmissionDirection.get() )
 {
  pSource->SetEnableFixedEmissionDirection( upCmdSetEnableFixedEmissionDirection->GetNewBoolValue( new_value ) );
 }
 else if ( command == upCmdSetEmissionEnergy.get() )
 {
  pSource->SetEmissionEnergy( upCmdSetEmissionEnergy->GetNewDoubleValue( new_value ) );
 }
 else if ( command == upCmdSetSeed.get() )
 {
  pSource->SetSeed( static_cast<G4long>( upCmdSetSeed->GetNewIntValue( new_value ) ) );
 }
 else if ( command == upCmdSetPositroniumLifetime.get() )
 {
  G4String positronium_name;
  G4String units;
  G4double value = 0.0;

  std::stringstream ss(new_value);
  ss >> positronium_name >> value >> units;

  G4String new_lifetime_value = std::to_string( value ) + " " + units;
  pSource->SetPostroniumLifetime(positronium_name, upCmdSetLifetime->GetNewDoubleValue(new_lifetime_value ));
  
 }
 else if ( command == upCmdSetPromptGammaEnergy.get() )
 {
  pSource->SetPromptGammaEnergy( upCmdSetPromptGammaEnergy->GetNewDoubleValue( new_value ) );
 }
 else if ( command ==  upCmdSetPositroniumFraction.get() )
 {
  G4double fraction = 0.0;
  G4String positronium_kind;

  std::stringstream ss(new_value);
  ss >> positronium_kind >> fraction;

  pSource->SetPositroniumFraction(positronium_kind, fraction);
 }
 else if(command ==  upCmdSetPositroniumFractions.get())
 {
  std::vector<float> fractions;
  std::stringstream ss(new_value);
  G4double num;
  while (ss >> num) {
    fractions.push_back(num);
  }
  fParamGenerator.SetPositroniumFraction(fractions);
  //pSource->fPositroniumFractions=fractions;
 }
 else if(command ==  upCmdSetPositroniumLifetimes.get())
 {
  std::vector<float> lifetimes;
  std::stringstream ss(new_value);
  G4double num;
  while (ss >> num) {
    lifetimes.push_back(num);
  }
  //pSource->fPositroniumLifetimes=lifetimes;
  fParamGenerator.SetPostroniumLifetimes(lifetimes);
 }
 else if(command ==  upCmdSetIsPromptPhoton.get())
 {
  std::vector<bool> isPromptPhoton;
  std::stringstream ss(new_value);
  bool flag;
  while (ss >> std::boolalpha >> flag) {
    isPromptPhoton.push_back(flag);
  }
  //pSource->fIsPromptPhoton=isPromptPhoton;
  fParamGenerator.SetEnableDeexcitation(isPromptPhoton);
 } 
 else if(command ==  upCmdSetPromptPhotonEnergies.get()) {
  std::vector<float> promptPhotonEnergies;
  std::stringstream ss(new_value);
  float energy;
  while (ss >> energy) {
    promptPhotonEnergies.push_back(energy);
  }
  //pSource->fPromptPhotonEnergies=promptPhotonEnergies;
  fParamGenerator.SetPromptGammaEnergies(promptPhotonEnergies);
 }
 else if(command ==  upCmdSetDecayKinds.get()) {
  std::vector<PositroniumDecayKind> decayKinds;
  std::stringstream ss(new_value);
  std::string kind;
  while (ss >> kind) {
    if (kind == "k2Gamma")  {
        decayKinds.push_back(k2Gamma);
      } else {
      decayKinds.push_back(k3Gamma);
      }
  }
  fParamGenerator.SetDecayKinds(decayKinds);
  //pSource->fDecayKinds=decayKinds;
 }
 else
 {
  GateVSourceMessenger::SetNewValue(command, new_value);
 }
}


PositroniumDecayModelParams GateExtendedVSourceMessenger::generatePositroniumDecayParams() const
{
  return fParamGenerator.generatePositroniumDecayParams();
}


