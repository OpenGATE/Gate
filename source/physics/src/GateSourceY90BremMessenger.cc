#include "GateSourceY90BremMessenger.hh" 
#include "GateSourceY90Brem.hh"
 
GateSourceY90BremMessenger::GateSourceY90BremMessenger(GateSourceY90Brem *source)
  : GateVSourceMessenger(source),
    mSource(source)
{
  G4String cmdName;

  cmdName = GetDirectoryName()+"setMinEnergy";
  setMinEnergy = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setMinEnergy->SetGuidance("Set the minimum energy of the bremsstrahlung generated");
  setMinEnergy->SetParameterName("min_energy", false);

}

GateSourceY90BremMessenger::~GateSourceY90BremMessenger()
{
  delete setMinEnergy;
}

void GateSourceY90BremMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  GateVSourceMessenger::SetNewValue(command,newValue);
  if ( command == setMinEnergy )
    mSource->SetMinEnergy(setMinEnergy->GetNewDoubleValue(newValue));
}


