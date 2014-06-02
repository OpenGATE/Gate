/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDiscretizer.hh"

#include "G4UnitsTable.hh"

#include "GateSystemListManager.hh"
#include "GateVSystem.hh"
#include "GateOutputVolumeID.hh"
#include "GateDiscretizerMessenger.hh"
#include "GateTools.hh"


GateDiscretizer::GateDiscretizer(GatePulseProcessorChain* itsChain,
      	      	      	 const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
{
  m_messenger = new GateDiscretizerMessenger(this);
}




GateDiscretizer::~GateDiscretizer()
{
  delete m_messenger;
}



void GateDiscretizer::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
    GateVSystem* system = GateSystemListManager::GetInstance()->GetSystem(0);
    G4ThreeVector pos = system->ComputeObjectCenter(&inputPulse->GetVolumeID());
    GatePulse* pulse = new GatePulse(*inputPulse);
    pulse->SetGlobalPos(pos);
    outputPulseList.push_back(pulse);
}





void GateDiscretizer::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Discretizer"  << G4endl;
}
