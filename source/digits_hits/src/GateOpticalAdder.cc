
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateOpticalAdder

*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#include "GateOpticalAdder.hh"
#include "GateOpticalAdderMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"



GateOpticalAdder::GateOpticalAdder(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {

	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateOpticalAdderMessenger(this);
}


GateOpticalAdder::~GateOpticalAdder()
{
  delete m_Messenger;

}


void GateOpticalAdder::Digitize()
{
	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;

  if (IDC)
     {
	  G4int n_digi = IDC->entries();
	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  if (inputDigi->IsOptical())
		    {
			  for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
			    {
				  if ( (*iter)->GetVolumeID()   == inputDigi->GetVolumeID() )
				  {
					  m_outputDigi= CentroidMerge( inputDigi, *iter );

					  if (nVerboseLevel>1)
						  G4cout << "Merged previous digi for volume " << inputDigi->GetVolumeID()
						  << " with new digi of energy " << G4BestUnit(inputDigi->GetEnergy(),"Energy") <<".\n"
						  << "Resulting digi is: \n"
						  << **iter << Gateendl << Gateendl ;
					  break;
				  }
			    }


			  if ( iter == OutputDigiCollectionVector->end() )
			  {
				  m_outputDigi = new GateDigi(*inputDigi);
				  if (nVerboseLevel>1)
					  G4cout << "Created new digi for volume " << inputDigi->GetVolumeID() << ".\n"
					  << "Resulting digi is: \n"
					  << *m_outputDigi << Gateendl << Gateendl ;
				  m_OutputDigiCollection->insert(m_outputDigi);
			  }

		    }
	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateOpticalAdder::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }

  StoreDigiCollection(m_OutputDigiCollection);

}


void GateOpticalAdder::DescribeMyself(size_t indent )
{
  ;
}
#endif
