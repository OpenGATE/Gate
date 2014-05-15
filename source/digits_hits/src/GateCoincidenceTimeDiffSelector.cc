/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidenceTimeDiffSelector.hh"
#include "G4UnitsTable.hh"
#include "GateCoincidenceTimeDiffSelectorMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"
#include "GateDetectorConstruction.hh"
#include "GateCrystalSD.hh"
#include "GateVSystem.hh"
#include "GateObjectChildList.hh"
#include "GateVVolume.hh"
#include "GateMaps.hh"


GateCoincidenceTimeDiffSelector::GateCoincidenceTimeDiffSelector(GateCoincidencePulseProcessorChain* itsChain,
			   const G4String& itsName)
  : GateVCoincidencePulseProcessor(itsChain,itsName)
{
  m_minTime = -1;
  m_maxTime = -1;

  m_messenger = new GateCoincidenceTimeDiffSelectorMessenger(this);
}




GateCoincidenceTimeDiffSelector::~GateCoincidenceTimeDiffSelector()
{
  delete m_messenger;
}




GateCoincidencePulse* GateCoincidenceTimeDiffSelector::ProcessPulse(GateCoincidencePulse* inputPulse,G4int )
{
  if (!inputPulse) {
      if (nVerboseLevel>1)
      	G4cout << "[GateCoincidenceTimeDiffSelector::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
      return 0;
  }

  G4double timeDiff = inputPulse->ComputeFinishTime()-inputPulse->GetStartTime();
  if ( ((m_minTime>0)  && (timeDiff<m_minTime) )
      ||
       ((m_maxTime>0)  && (timeDiff>m_maxTime) ) )
       return 0;
  else
       return new GateCoincidencePulse(*inputPulse);
}


void GateCoincidenceTimeDiffSelector::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "TimeDiffSelector: "
      	 << G4BestUnit(m_minTime,"Time") << "/"<< G4BestUnit(m_minTime,"Time")  << G4endl;
}
