/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTransferEfficiency.hh"

#include "GateTransferEfficiencyMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "G4VSolid.hh"
#include "G4Box.hh"

#include "G4UnitsTable.hh"

#include "GateObjectStore.hh"


// Static pointer to the GateTransferEfficiency singleton
GateTransferEfficiency* GateTransferEfficiency::theGateTransferEfficiency=0;



/*    	This function allows to retrieve the current instance of the GateTransferEfficiency singleton
      	If the GateTransferEfficiency already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateTransferEfficiency constructor
*/
GateTransferEfficiency* GateTransferEfficiency::GetInstance(GatePulseProcessorChain* itsChain,
			       const G4String& itsName)
{
  if (!theGateTransferEfficiency)
    if (itsChain)
      theGateTransferEfficiency = new GateTransferEfficiency(itsChain, itsName);
  return theGateTransferEfficiency;
}


// Private constructor
GateTransferEfficiency::GateTransferEfficiency(GatePulseProcessorChain* itsChain,
			       const G4String& itsName)
  : GateVPulseProcessor(itsChain, itsName)
{
  m_messenger = new GateTransferEfficiencyMessenger(this);
}

// Public destructor
GateTransferEfficiency::~GateTransferEfficiency()
{
  delete m_messenger;
}

G4int GateTransferEfficiency::ChooseVolume(G4String val)
{
  //Retrieve the inserter store to check if the volume name is valid
  GateObjectStore* m_store = GateObjectStore::GetInstance();
  if (m_store->FindCreator(val)!=0) {
    m_table[val]=1.;
    return 1;
  }
  else {
    G4cout << "Wrong Volume Name" << G4endl;
    return 0;
  }
}

void GateTransferEfficiency::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
  GatePulse* outputPulse = new GatePulse(*inputPulse);
  if(im != m_table.end())
    {
      if(((*im).second < 0) | ((*im).second > 1)) {
	G4cerr << 	G4endl << "[GateLightYield::ProcessOnePulse]:" << G4endl
	       <<   "Sorry, but the transfer efficiency (" << (*im).second << ") for "
	       << (*im).first << " is invalid" << G4endl;
	G4String msg = "It must be a number between 0 and 1 !!!\n"
        "You must set the transfer efficiency:\n"
        "\t/gate/digitizer/Singles/transferEfficiency/" +
        (*im).first + "/setTECoef NUMBER\n"
        "or disable the transfer efficiency module using:\n"
        "\t/gate/digitizer/Singles/transferEfficiency/disable\n";
	G4Exception( "GateTransferEfficiency::ProcessOnePulse", "ProcessOnePulse", FatalException, msg );
      }
      else
	{
	  m_TECoef = (*im).second;
	  outputPulse->SetEnergy( m_TECoef * inputPulse->GetEnergy() );
	}
    }
  outputPulseList.push_back(outputPulse);
}

G4double GateTransferEfficiency::GetTEMin()
{
  im=m_table.begin();
  m_TEMin = (*im).second;
  for (im=m_table.begin(); im!=m_table.end(); im++)
    m_TEMin = ((*im).second <= m_TEMin) ?
      (*im).second : m_TEMin;
  return m_TEMin;
}

void GateTransferEfficiency::DescribeMyself(size_t indent)
{
  for (im=m_table.begin(); im!=m_table.end(); im++)
    G4cout << GateTools::Indent(indent) << (*im).first << ":\n"
	   << GateTools::Indent(indent+1) << "Transfer Efficiency: " << (*im).second << G4endl;
}
