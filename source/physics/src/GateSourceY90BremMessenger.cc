#include "GateSourceY90BremMessenger.hh" 
#include "GateSourceY90Brem.hh"
 
GateSourceY90BremMessenger::GateSourceY90BremMessenger(GateSourceY90Brem *source)
  : GateVSourceMessenger(source),
    mSource(source)
{
  G4String cmdName;

  cmdName = GetDirectoryName()+"setMinBremEnergy";
  setMinBremEnergyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setMinBremEnergyCmd->SetGuidance("Set the minimum energy of the bremsstrahlung generated");
  setMinBremEnergyCmd->SetParameterName("min_energy", false);

  cmdName = GetDirectoryName()+"setPositronProbability";
  setPosProbabilityCmd = new G4UIcmdWithADouble(cmdName,this);
  setPosProbabilityCmd->SetGuidance("Set the likelihood of positron production. Default is 3.184e-5.");
  setPosProbabilityCmd->SetParameterName("pos_probability", false);
}

GateSourceY90BremMessenger::~GateSourceY90BremMessenger()
{
  delete setMinBremEnergyCmd;
  delete setPosProbabilityCmd;
}

void GateSourceY90BremMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  GateVSourceMessenger::SetNewValue(command,newValue);
  if ( command == setMinBremEnergyCmd )
    mSource->SetMinEnergy(setMinBremEnergyCmd->GetNewDoubleValue(newValue));
  else if (command == setPosProbabilityCmd)
    mSource->SetPositronProbability(setPosProbabilityCmd->GetNewDoubleValue(newValue));
}


