/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateCoincidenceDigiMaker.hh"

#include "G4DigiManager.hh"

#include "GateCoincidenceDigi.hh"
#include "GateDigitizer.hh"
#include "GateTools.hh"
#include "GateDigitizer.hh"
#include "GateOutputMgr.hh"

// Constructor
GateCoincidenceDigiMaker::GateCoincidenceDigiMaker( GateDigitizer* itsDigitizer,
      	      	         	        	    const G4String& itsInputName,
						    G4bool itsOutputFlag)
  :GateVDigiMakerModule(itsDigitizer,itsInputName)
{
  GateOutputMgr::GetInstance()->RegisterNewCoincidenceDigiCollection( GetCollectionName(),itsOutputFlag );
}



// Destructor
GateCoincidenceDigiMaker::~GateCoincidenceDigiMaker()
{
}




// Convert a pulse list into a Coincidence Digi collection
void GateCoincidenceDigiMaker::Digitize()
{
  std::vector<GateCoincidencePulse*> coincidencePulse = GateDigitizer::GetInstance()->FindCoincidencePulse(m_inputName);
  if (coincidencePulse.empty()) {
    if (nVerboseLevel)
      G4cout  << "[GateCoincidenceDigiMaker::Digitize]: coincidence pulse null --> no digi created\n";
    return;
  }
  // Create the digi collection
  GateCoincidenceDigiCollection* CoincidenceDigiCollection = new GateCoincidenceDigiCollection(m_digitizer->GetObjectName(),m_collectionName);

  // Create and store the digi
  for (std::vector<GateCoincidencePulse*>::const_iterator it = coincidencePulse.begin();it != coincidencePulse.end() ; ++it){
    // removed by OK 12/01/23
    /* if ((*it)->size()>2){
      	if (nVerboseLevel)
      	    G4cout  << "[GateCoincidenceDigiMaker::Digitize]: ignoring multiple coincidence --> no digits created\n";
      } else {
    */
      	GateCoincidenceDigi* Digi = new GateCoincidenceDigi( **it);
      	CoincidenceDigiCollection->insert(Digi);
	// }
  }

  if (nVerboseLevel>0) {
      	  G4cout  << "[GateCoincidenceDigiMaker::Digitize]:  created 1 coincidence digi in this event:" <<	Gateendl;
	  (*CoincidenceDigiCollection)[0]->Print();
	  G4cout << Gateendl;
  }


  // Store the digits into the digit collection of this event
  m_digitizer->StoreDigiCollection(CoincidenceDigiCollection);
}
