/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTimestampDiscretizer.hh"
#include "G4UnitsTable.hh"
#include "GateTimestampDiscretizerMessenger.hh"
#include "GateTools.hh"

GateTimestampDiscretizer::GateTimestampDiscretizer(GatePulseProcessorChain* itsChain,
			       const G4String& itsName,
						 G4double itsSamplingtime)
  : GateVPulseProcessor(itsChain,itsName),
		m_samplingtime(itsSamplingtime)
{
  m_messenger = new GateTimestampDiscretizerMessenger(this);
}


GateTimestampDiscretizer::~GateTimestampDiscretizer()
{
  delete m_messenger;
}



void GateTimestampDiscretizer::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if (!inputPulse) {
    if (nVerboseLevel>1) {
      	G4cout << "[GateTimestampDiscretizer::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
    }
    return;
  } else {
  	GatePulse* outputPulse = new GatePulse(*inputPulse);
  	G4double time = floor(inputPulse->GetTime() / m_samplingtime) * m_samplingtime;
  	outputPulse->SetTime(time);
  	outputPulseList.push_back(outputPulse);
  	if (nVerboseLevel>1) {
  		G4cout << "Pulse real time: \n"
  				<< G4BestUnit(inputPulse->GetTime(),"Time") << Gateendl
					<< "Discretized pulse time: \n"
					<< G4BestUnit(outputPulse->GetTime(),"Time") << Gateendl;
  	}
  }
}



void GateTimestampDiscretizer::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Sampling time: " << G4BestUnit(m_samplingtime,"Time") << "( " << G4BestUnit(1.0/m_samplingtime,"Frequency") << " )" << Gateendl;
}
