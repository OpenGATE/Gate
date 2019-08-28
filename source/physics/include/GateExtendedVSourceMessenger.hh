
#ifndef GateExtendedVSourceMessenger_hh
#define GateExtendedVSourceMessenger_hh

#include "GateVSourceMessenger.hh"
#include "G4UImessenger.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"

class GateExtendedVSource;

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  About class: Messenger for GateExtendedVSource class
 * */
class GateExtendedVSourceMessenger: public GateVSourceMessenger
{
 public:
  GateExtendedVSourceMessenger( GateExtendedVSource* source );
  ~GateExtendedVSourceMessenger();

  void SetNewValue( G4UIcommand* command, G4String newValue );

 protected:
  void InitCommands();
  void DeleteCommands();

 protected:
  GateExtendedVSource* pSource = nullptr;
  G4UIcmdWithAnInteger* pCmdSeedForRandomGenerator = nullptr;
  G4UIcmdWithADoubleAndUnit* pCmdPromptGammaEnergy = nullptr;
  G4UIcmdWithADoubleAndUnit* pCmdLinearPolarization = nullptr;
  G4UIcmdWithABool* pCmdUseUnpolarizedParticles = nullptr;
};

#endif
