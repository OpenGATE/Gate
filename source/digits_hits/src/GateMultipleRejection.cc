/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateMultipleRejection

  Digitizer module for simulating a MultipleRejection

  December 2023: rewritten by olga.kochebina@cea.fr
  Important ! Not all options are tested

*/

#include "GateMultipleRejection.hh"
#include "GateMultipleRejectionMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "Randomize.hh"
#include "GateObjectStore.hh"
#include "GateConstants.hh"



GateMultipleRejection::GateMultipleRejection(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_MultipleRejection(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateMultipleRejectionMessenger(this);

	//GateDigiCollection* tempDigiCollection = new GateDigiCollection(GetName(), m_digitizer-> GetOutputName()); // to create the Digi Collection
	//m_waiting = tempDigiCollection->GetVector ();
}


GateMultipleRejection::~GateMultipleRejection()
{
  delete m_Messenger;

}


void GateMultipleRejection::Digitize()
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
	std::vector<std::vector<GateDigi*>::iterator> toDel;

  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  m_outputDigi = new GateDigi(*inputDigi);

		  G4String currentVolumeName = (inputDigi->GetVolumeID().GetBottomCreator())->GetObjectName();
		  GateVolumeID currentVolumeID = inputDigi->GetVolumeID();
		  m_VolumeNames.push_back(currentVolumeName);
		  m_VolumeIDs.push_back(currentVolumeID);

		  if(i>1)
		  {
			if(m_multipleDef==kvolumeID)
			{
				if ( std::find(m_VolumeIDs.begin(), m_VolumeIDs.end(), currentVolumeID) != m_VolumeIDs.end() )
					return;
				else
					m_OutputDigiCollection->insert(m_outputDigi);
			}
			else
			{
				if (std::find(m_VolumeNames.begin(), m_VolumeNames.end(), currentVolumeName) != m_VolumeNames.end() )
					return;

			}
		  }
		else
			{
				if (n_digi==1) //save if only one digi
					m_OutputDigiCollection->insert(m_outputDigi);

			}

	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateMultipleRejection::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);


}

void GateMultipleRejection::DescribeMyself(size_t indent )
{

	    G4cout << GateTools::Indent(indent) << "Multiple rejection " << m_digitizer->GetSD()->GetName() << ":\n"
	       << GateTools::Indent(indent+1) << m_multipleDef <<
	         GateTools::Indent(indent+1) << m_MultipleRejection<<Gateendl;

}
