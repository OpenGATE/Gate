/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateExtendedVSourceMessenger_hh
#define GateExtendedVSourceMessenger_hh

#include "GateVSourceMessenger.hh"
#include "G4UImessenger.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include <memory>

class GateExtendedVSource;

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Messenger for GateExtendedVSource class
 **/
class GateExtendedVSourceMessenger: public GateVSourceMessenger
{
 public:
  GateExtendedVSourceMessenger( GateExtendedVSource* source );
  ~GateExtendedVSourceMessenger();

  void SetNewValue( G4UIcommand* command, G4String newValue );

 protected:
  void InitCommands();
  G4UIcmdWithABool* GetBoolCmd( const G4String cmd_name, const G4String cmd_guidance );
  G4UIcmdWithADoubleAndUnit* GetDoubleCmdWithUnit( const G4String cmd_name, const G4String cmd_guidance, const G4String default_unit, const G4String unit_candidates );
  G4UIcmdWith3Vector* GetVectorCmd( const G4String cmd_name, const G4String cmd_guidance );
  G4UIcmdWithAnInteger* GetIntCmd( const G4String cmd_name, const G4String cmd_guidance );
  G4UIcmdWithAString* GetStringCmd(const G4String cmd_name, const G4String cmd_guidance );
  G4UIcmdWith3VectorAndUnit* GetVectorCmdWithUnit( const G4String cmd_name, const G4String cmd_guidance, const G4String default_unit, const G4String unit_candidates );

 protected:
  GateExtendedVSource* pSource = nullptr;
  std::unique_ptr<G4UIcmdWithABool> upCmdSetEnableDeexcitation;
  std::unique_ptr<G4UIcmdWith3Vector> upCmdSetFixedEmissionDirection;
  std::unique_ptr<G4UIcmdWithABool> upCmdSetEnableFixedEmissionDirection;
  std::unique_ptr<G4UIcmdWithADoubleAndUnit> upCmdSetEmissionEnergy;
  std::unique_ptr<G4UIcmdWithAnInteger> upCmdSetSeed;
  std::unique_ptr<G4UIcmdWithAString> upCmdSetPostroniumLifetime;
  std::unique_ptr<G4UIcmdWithADoubleAndUnit> upCmdSetPromptGammaEnergy;
  std::unique_ptr<G4UIcmdWithAString> upCmdSetPositroniumFraction;
  //Supporting commands - disable for user
  std::unique_ptr<G4UIcmdWithADoubleAndUnit> upCmdSetLifetime;
  
};

#endif
