/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidenceGeometrySelector.hh"
#include "G4UnitsTable.hh"
#include "GateCoincidenceGeometrySelectorMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"
#include "GateVVolume.hh"
#include "GateDetectorConstruction.hh"
#include "GateCrystalSD.hh"
#include "GateVSystem.hh"
#include "GateObjectChildList.hh"
#include "GateMaps.hh"




GateCoincidenceGeometrySelector::GateCoincidenceGeometrySelector(GateCoincidencePulseProcessorChain* itsChain,
			   const G4String& itsName)
  : GateVCoincidencePulseProcessor(itsChain,itsName)
{
  m_maxS = -1;
  m_maxDeltaZ = -1;

  m_messenger = new GateCoincidenceGeometrySelectorMessenger(this);
}




GateCoincidenceGeometrySelector::~GateCoincidenceGeometrySelector()
{
  delete m_messenger;
}




GateCoincidencePulse* GateCoincidenceGeometrySelector::ProcessPulse(GateCoincidencePulse* inputPulse,G4int )
{
  if (!inputPulse) {
      if (nVerboseLevel>1)
      	G4cout << "[GateCoincidenceGeometrySelector::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
      return 0;
  }

  G4double s;
  if (inputPulse->size() != 2) {
      if (nVerboseLevel>1)
      	G4cout << "[GateCoincidenceGeometrySelector::ProcessOnePulse]: input pulse has not 2 pulses -> nothing to do\n\n";
      return 0;
  }

  G4ThreeVector globalPos1 = (*inputPulse)[0]->GetGlobalPos();
  G4ThreeVector globalPos2 = (*inputPulse)[1]->GetGlobalPos();

  if ((m_maxDeltaZ>0) && (fabs(globalPos2.z()-globalPos1.z())>m_maxDeltaZ) )
   return 0;

  G4double denom = (globalPos1.y()-globalPos2.y()) * (globalPos1.y()-globalPos2.y()) +
                   (globalPos2.x()-globalPos1.x()) * (globalPos2.x()-globalPos1.x());

  if (denom!=0.) {
    denom = sqrt(denom);

    s = ( globalPos1.x() * (globalPos2.y()-globalPos1.y()) +
	  globalPos1.y() * (globalPos1.x()-globalPos2.x())  )
      	/ denom;
  } else {
    s = 0.;
  }

  G4double theta;
  if ((globalPos1.x()-globalPos2.x())!=0.) {
    theta=atan((globalPos1.x()-globalPos2.x()) /
	       (globalPos1.y()-globalPos2.y()));
  } else {
    theta=3.1416/2.;
  }
  if ((theta > 0.) && ((globalPos1.x()-globalPos2.x()) > 0.)) s = -s;
  if ((theta < 0.) && ((globalPos1.x()-globalPos2.x()) < 0.)) s = -s;
  if ( theta < 0.) {
    theta = theta+3.1416;
    s = -s;
  }
  if ((m_maxS>0)  && (fabs(s)>m_maxS) ) return 0;

  return new GateCoincidencePulse(*inputPulse);
}


void GateCoincidenceGeometrySelector::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "GeometrySelector: "
      	 << "SMax      : "<<G4BestUnit(m_maxS,"Length")
      	 << "DeltaZMax : "<<G4BestUnit(m_maxDeltaZ,"Length") << G4endl;
}
