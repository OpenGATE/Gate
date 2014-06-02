/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTemporalResolution.hh"

#include "G4UnitsTable.hh"

#include "GateTemporalResolutionMessenger.hh"
#include "GateTools.hh"

#include "Randomize.hh"


GateTemporalResolution::GateTemporalResolution(GatePulseProcessorChain* itsChain,
					       const G4String& itsName,
					       G4double itsTimeResolution)
  : GateVPulseProcessor(itsChain, itsName),
    m_timeResolution(itsTimeResolution)
{
  m_messenger = new GateTemporalResolutionMessenger(this);
}




GateTemporalResolution::~GateTemporalResolution()
{
  delete m_messenger;
}



void GateTemporalResolution::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if(m_timeResolution < 0 ) {
    G4cerr << 	G4endl << "[GateTemporalResolution::ProcessOnePulse]:" << G4endl
      	   <<   "Sorry, but the resolution (" << GetTimeResolution() << ") is invalid" << G4endl;
    G4Exception( "GateTemporalResolution::ProcessOnePulse", "ProcessOnePulse", FatalException,
			"You must choose a temporal resolution >= 0 /gate/digitizer/Singles/Singles/timeResolution/setTimeResolution TIME\n or disable the temporal resolution using:\n\t/gate/digitizer/Singles/Singles/timeResolution/disable\n");
  }
  else {
    if (!inputPulse) {
      if (nVerboseLevel>1)
	G4cout << "[GateTemporalResolution::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
      return;
    }

    if (inputPulse)
      {
	GatePulse* outputPulse = new GatePulse(*inputPulse);
	// set the new time by a Gaussian shot of mean: old time, and of sigma: m_timeResolution/2.35
	G4double sigma =  m_timeResolution / 2.35;
	outputPulse->SetTime(G4RandGauss::shoot(inputPulse->GetTime(), sigma));
	outputPulseList.push_back(outputPulse);

	if (nVerboseLevel>1)
	  {
	    G4cout << "Pulse real time: " << G4endl
		   << G4BestUnit(inputPulse->GetTime(),"Time") << G4endl
		   << "Pulse new time: " << G4endl
		   << G4BestUnit(outputPulse->GetTime(),"Time") << G4endl
		   << "Difference (real - new time): " << G4endl
		   << G4BestUnit(inputPulse->GetTime() - outputPulse->GetTime(),"Time")
		   << G4endl << G4endl ;

	  }
      }
  }
}


void GateTemporalResolution::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Temporal resolution: " << G4BestUnit(m_timeResolution,"Time") << G4endl;
}
