/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSpblurring.hh"

#include "G4ThreeVector.hh"

#include "G4UnitsTable.hh"

#include "GateSpblurringMessenger.hh"
#include "GateTools.hh"

#include "Randomize.hh"

GateSpblurring::GateSpblurring(GatePulseProcessorChain* itsChain,
			       const G4String& itsName,
      	      	      	       G4double itsSpresolution)
  : GateVPulseProcessor(itsChain,itsName),
    m_spresolution(itsSpresolution)
{
  m_messenger = new GateSpblurringMessenger(this);
}




GateSpblurring::~GateSpblurring()
{
  delete m_messenger;
}



void GateSpblurring::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{

	GatePulse* outputPulse = new GatePulse(*inputPulse);
	//TC G4ThreeVector P = inputPulse->GetGlobalPos();
	G4ThreeVector P = inputPulse->GetVolumeID().MoveToBottomVolumeFrame(inputPulse->GetGlobalPos()); //TC
	G4double Px = P.x();
	G4double Py = P.y();
	G4double Pz = P.z();
	G4double PxNew = G4RandGauss::shoot(Px,m_spresolution/2.35);
	G4double PyNew = G4RandGauss::shoot(Py,m_spresolution/2.35);
	G4double PzNew = G4RandGauss::shoot(Pz,m_spresolution/2.35); //TC
	//TC G4double PzNew = Pz;
	outputPulse->SetLocalPos(G4ThreeVector(PxNew,PyNew,PzNew)); //TC
	outputPulse->SetGlobalPos(outputPulse->GetVolumeID().MoveToAncestorVolumeFrame(outputPulse->GetLocalPos())); //TC
	//TC outputPulse->SetGlobalPos(G4ThreeVector(PxNew,PyNew,PzNew));
	outputPulseList.push_back(outputPulse);
}

void GateSpblurring::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Resolution : " << m_spresolution  << G4endl;
}
