/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateLocalBlurring.hh"

#include "GateLocalBlurringMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"

#include "Randomize.hh"

#include "G4UnitsTable.hh"

#include "GateObjectStore.hh"


GateLocalBlurring::GateLocalBlurring(GatePulseProcessorChain* itsChain,
				     const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
{
  m_messenger = new GateLocalBlurringMessenger(this);
}

GateLocalBlurring::~GateLocalBlurring()
{
  delete m_messenger;
}

G4int GateLocalBlurring::ChooseVolume(G4String val)
{
  GateObjectStore* m_store = GateObjectStore::GetInstance();

  if (m_store->FindCreator(val)!=0) {
    m_param.resolution = -1;
    m_param.eref = -1;
    m_table[val] = m_param;
    return 1;
  }
  else {
    G4cout << "Wrong Volume Name" << G4endl;
    return 0;
  }
}

void GateLocalBlurring::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
  GatePulse* outputPulse = new GatePulse(*inputPulse);
  if(im != m_table.end())
    {
      if((*im).second.resolution < 0 ) {
	G4cerr << 	G4endl << "[GateLocalBlurring::ProcessOnePulse]:" << G4endl
	       <<   "Sorry, but the resolution (" << (*im).second.resolution << ") for " << (*im).first << " is invalid" << G4endl;
	G4String msg = "You must set the energy of reference AND the resolution: /gate/digitizer/Singles/localBlurring/" + (*im).first + "/setEnergyOfReference ENERGY or disable the local blurring using: /gate/digitizer/Singles/localBlurring/disable\n";
	G4Exception( "GateLocalBlurring::ProcessOnePulse", "ProcessOnePulse", FatalException, msg );
      }
      else if((*im).second.eref < 0) {
	G4cerr <<   G4endl << "[GateLocalBlurring::ProcessOnePulse]:" << G4endl
	       <<   "Sorry, but the energy of reference (" << G4BestUnit((*im).second.eref,"Energy") << ") for "
	       << (*im).first <<" is invalid" << G4endl;
	G4String msg = "You must set the resolution AND the energy of reference:\n\t/gate/digitizer/Singles/localBlurring/" + (*im).first + "/setEnergyOfReference ENERGY\nor disable the local blurring using:\n\t/gate/digitizer/Singles/localBlurring/disable";
	G4Exception( "GateLocalBlurring::ProcessOnePulse", "ProcessOnePulse", FatalException, msg );
	}
      else {
	G4double m_coeff = (*im).second.resolution * sqrt((*im).second.eref);
	outputPulse->SetEnergy(G4RandGauss::shoot(inputPulse->GetEnergy(),m_coeff*sqrt(inputPulse->GetEnergy())/2.35));
      }
    }
  outputPulseList.push_back(outputPulse);
}

void GateLocalBlurring::DescribeMyself(size_t indent)
{
  for (im=m_table.begin(); im!=m_table.end(); im++)
    G4cout << GateTools::Indent(indent) << "Resolution of " << (*im).first << ":\n"
	   << GateTools::Indent(indent+1) << (*im).second.resolution << "  for "
	   << G4BestUnit((*im).second.eref,"Energy") <<  G4endl;
}
