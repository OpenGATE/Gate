/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSingleDigiMaker.hh"

#include "G4DigiManager.hh"

#include "GateSingleDigi.hh"
#include "GateTools.hh"
#include "GateDigitizer.hh"
#include "GateOutputMgr.hh"

// Constructor
GateSingleDigiMaker::GateSingleDigiMaker( GateDigitizer* itsDigitizer,
      	      	         	          const G4String& itsInputName,
					  G4bool itsOutputFlag)
  :GateVDigiMakerModule(itsDigitizer,itsInputName)
{
//  G4cout << " in GateSingleDigiMaker call RegisterNewSingleDigiCollection"  << G4endl;
  GateOutputMgr::GetInstance()->RegisterNewSingleDigiCollection( GetCollectionName(),itsOutputFlag );
}



// Destructor
GateSingleDigiMaker::~GateSingleDigiMaker()
{
}




// Convert a pulse list into a single Digi collection
void GateSingleDigiMaker::Digitize()
{
  if (nVerboseLevel>1)
    G4cout  << "[GateSingleDigiMaker::Digitize]: retrieving pulse-list '" << m_inputName << "'" << G4endl;

  GatePulseList* pulseList = GateDigitizer::GetInstance()->FindPulseList(m_inputName);

  if (!pulseList) {
    if (nVerboseLevel>1)
      G4cout  << "[GateSingleDigiMaker::Digitize]: pulse list null --> no digits created\n";
    return;
  }

  if (pulseList->empty()) {
    if (nVerboseLevel>1)
      G4cout  << "[GateSingleDigiMaker::Digitize]: pulse list empty --> no digits created\n";
    return;
  }

  // Get number of pulses from hit processing
  size_t n_pulses = pulseList->size();
  size_t i;

  // Create the digi collection
  GateSingleDigiCollection* singleDigiCollection = new GateSingleDigiCollection(m_digitizer->GetObjectName(),m_collectionName);

  // Transform each pulse into a single digi
  for (i=0;i<n_pulses;i++) {
	GateSingleDigi* Digi = new GateSingleDigi( (*pulseList)[i] );
	singleDigiCollection->insert(Digi);
      }

  if (nVerboseLevel>0) {
      	G4cout  << "[GateSingleDigiMaker::ConvertSinglePulseList]: "
	      	<< "created " << singleDigiCollection->entries() << " single digits:" << G4endl;
      	for (i=0; i<(size_t)(singleDigiCollection->entries()); i++)
	  (*singleDigiCollection)[i]->Print();
	G4cout << G4endl;
  }

  // Store the digits into the digit collection of this event
  m_digitizer->StoreDigiCollection(singleDigiCollection);
}
