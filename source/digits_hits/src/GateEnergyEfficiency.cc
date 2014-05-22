/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateEnergyEfficiency.hh"

#include "GateEnergyEfficiencyMessenger.hh"
#include "GateTools.hh"
#include "GateVSystem.hh"
#include "GateVDistribution.hh"
#include "GateSystemListManager.hh"

#include "Randomize.hh"

#include "G4UnitsTable.hh"
#include <fstream>



GateEnergyEfficiency::GateEnergyEfficiency(GatePulseProcessorChain* itsChain,
      	      	      	      	 const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName),
    m_efficiency(0)
{
  m_messenger = new GateEnergyEfficiencyMessenger(this);
}




GateEnergyEfficiency::~GateEnergyEfficiency()
{
  delete m_messenger;
}

void GateEnergyEfficiency::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
   if (!m_efficiency){ // default efficiency is 1
      outputPulseList.push_back(new GatePulse(*inputPulse));
      return;
   }
   GateVSystem* system = GateSystemListManager::GetInstance()->GetSystem(0);
   if (!system){
      G4cerr<<"[GateEnergyEfficiency::ProcessOnePulse] Problem : no system defined"<<G4endl;
      return ;
   }
   G4double eff = m_efficiency->Value(inputPulse->GetEnergy());
//   G4cout<<inputPulse->GetEnergy()<<"   "<<eff<<G4endl;
   if (G4UniformRand() < eff)
      outputPulseList.push_back(new GatePulse(*inputPulse));

}

void GateEnergyEfficiency::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Tabular Efficiency "<< G4endl;
}
