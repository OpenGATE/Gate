/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateTemporalResolution.hh"

#include "G4UnitsTable.hh"

#include "GateTemporalResolutionMessenger.hh"
#include "GateTools.hh"

#include "Randomize.hh"
#include "GateConstants.hh"


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
    G4cerr << 	Gateendl << "[GateTemporalResolution::ProcessOnePulse]:\n"
      	   <<   "Sorry, but the resolution (" << GetTimeResolution() << ") is invalid\n";
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
	G4double sigma =  m_timeResolution / GateConstants::fwhm_to_sigma;
	outputPulse->SetTime(G4RandGauss::shoot(inputPulse->GetTime(), sigma));
	outputPulseList.push_back(outputPulse);

	if (nVerboseLevel>1)
	  {
	    G4cout << "Pulse real time: \n"
		   << G4BestUnit(inputPulse->GetTime(),"Time") << Gateendl
		   << "Pulse new time: \n"
		   << G4BestUnit(outputPulse->GetTime(),"Time") << Gateendl
		   << "Difference (real - new time): \n"
		   << G4BestUnit(inputPulse->GetTime() - outputPulse->GetTime(),"Time")
		   << Gateendl << Gateendl ;

	  }
      }
  }
}


void GateTemporalResolution::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Temporal resolution: " << G4BestUnit(m_timeResolution,"Time") << Gateendl;
}
