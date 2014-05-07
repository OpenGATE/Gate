/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateNoiseMessenger.hh"
#include "GateNoise.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "GateVDistribution.hh"
#include "GateDistributionListManager.hh"

GateNoiseMessenger::GateNoiseMessenger(GateNoise* itsPulseProcessor)
    : GatePulseProcessorMessenger(itsPulseProcessor)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setDeltaTDistribution";
  m_deltaTDistribCmd = new G4UIcmdWithAString(cmdName,this);;
  m_deltaTDistribCmd->SetGuidance("Set the deltaT distribution");

  cmdName = GetDirectoryName() + "setEnergyDistribution";
  m_energyDistribCmd = new G4UIcmdWithAString(cmdName,this);;
  m_energyDistribCmd->SetGuidance("Set the energy distribution");

}


GateNoiseMessenger::~GateNoiseMessenger()
{
  delete m_deltaTDistribCmd;
  delete m_energyDistribCmd;
}


void GateNoiseMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  GateNoise* noise = static_cast<GateNoise*>(GetNamedObject());
  if (!noise) return;
  if ( command==m_deltaTDistribCmd ){
    GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
    if (distrib) noise->SetDeltaTDistribution(distrib);
  } else if (command==m_energyDistribCmd ){
    GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
    if (distrib) noise->SetEnergyDistribution(distrib);
  }
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
