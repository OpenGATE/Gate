/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateDigitizerInitializationModule
  This class is a specific digitizer class that are called before running all users
  digitizers and digitizer modules.
  It creates Digi from Hits of this event and fills/copies all attributes for this Digi

	05/2022 Olga.Kochebina@cea.fr
*/


#include "GateDigitizerInitializationModule.hh"
#include "GateDigi.hh"
#include "GateCrystalSD.hh"

#include "GateHit.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "GateDigitizerMgr.hh"

GateDigitizerInitializationModule::GateDigitizerInitializationModule(GateSinglesDigitizer *digitizer)
  :GateVDigitizerModule("DigiInit","digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/digiInit",digitizer, digitizer->GetSD()),
   m_FirstEvent(true),
   m_HCID(-1),
   m_outputDigiCollection(0),
   m_digitizer(digitizer)
{

	G4String colName = digitizer->GetOutputName();
	collectionName.push_back(colName);
}


GateDigitizerInitializationModule::~GateDigitizerInitializationModule()
{
	delete  m_digitizer;
}


void GateDigitizerInitializationModule::Digitize()
{

	G4cout<<"GateDigitizerInitializationModule::Digitize()"<<G4endl;

	m_outputDigiCollection = new GateDigiCollection (GetName(),  m_digitizer->GetOutputName() ); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();
	G4String HCname=m_digitizer->m_SD->GetName()+"Collection" ;



	if (m_FirstEvent)
	{
		m_HCID= DigiMan->GetHitsCollectionID(HCname);
		m_FirstEvent=false;
	}


	GateHitsCollection* inHC;
	GateDigitizerMgr* digitizerMgr=GateDigitizerMgr::GetInstance();


	if (GateDigitizerMgr::GetInstance()->IsOfflineMode() &&
			digitizerMgr->GetOfflineHitsCollection() == nullptr)
	{
	    GateMessage("OfflineDigi", 1,
	        "No offline hits, skipping digitization.");
	    return;
	}


	if (GateDigitizerMgr::GetInstance()->IsOfflineMode())
	    inHC = digitizerMgr->GetOfflineHitsCollection();
	else
	    inHC = (GateHitsCollection*)DigiMan->GetHitsCollection(m_HCID);

	G4cout << "Retrieved HC = " << inHC
	       << " entries = " << inHC->entries()
	       << G4endl;




	if (inHC)
    {
      G4int n_hit = inHC->entries();

	 for (G4int i=0;i<n_hit;i++)
	{
		 GateHit *hit = (*inHC)[i];
		 G4cout<<"Hit eventID "<<hit->GetEventID() <<G4endl;


    	 if(hit->GetEdep() !=0 )
    	  {
    		  GateDigi* Digi = new GateDigi();
    		  Digi->SetMother( hit );
    		  Digi->SetRunID( hit->GetRunID() );
    		  Digi->SetEventID( hit->GetEventID() );
    		  Digi->SetTrackID( hit->GetTrackID() );
    		  Digi->SetSourceID( hit->GetSourceID() );
    		  Digi->SetSourcePosition( hit->GetSourcePosition() );
    		  Digi->SetTime( hit->GetTime() );
    		  Digi->SetEnergy( hit->GetEdep() );
    		  Digi->SetMaxEnergy( hit->GetEdep() );
    		  Digi->SetLocalPos( hit->GetLocalPos() );
    		  Digi->SetGlobalPos( hit->GetGlobalPos() );
    		  Digi->SetPDGEncoding( hit->GetPDGEncoding() );
    		  Digi->SetOutputVolumeID( hit->GetOutputVolumeID() );
    		  Digi->SetNPhantomCompton( hit->GetNPhantomCompton() );
    		  Digi->SetNCrystalCompton( hit->GetNCrystalCompton() );
    		  Digi->SetNPhantomRayleigh( hit->GetNPhantomRayleigh() );
    		  Digi->SetNCrystalRayleigh( hit->GetNCrystalRayleigh() );
    		  Digi->SetComptonVolumeName( hit->GetComptonVolumeName() );
    		  Digi->SetRayleighVolumeName( hit->GetRayleighVolumeName() );
    		  Digi->SetVolumeID( hit->GetVolumeID() );
    		  Digi->SetSystemID( hit->GetSystemID() );
    		  Digi->SetScannerPos( hit->GetScannerPos() );
    		  Digi->SetScannerRotAngle( hit->GetScannerRotAngle() );
    		  #ifdef GATE_USE_OPTICAL
    		    Digi->SetOptical( hit->GetPDGEncoding() == -22);
    		  #endif
    		  Digi->SetNSeptal( hit->GetNSeptal() );  // HDS : septal penetration

    		  // AE : Added for IdealComptonPhot adder which take into account several Comptons in the same volume
    		  Digi->SetPostStepProcess(hit->GetPostStepProcess());
    		  Digi->SetEnergyIniTrack(hit->GetEnergyIniTrack());
    		  Digi->SetEnergyFin(hit->GetEnergyFin());
    		  Digi->SetProcessCreator(hit->GetProcess());
    		  Digi->SetTrackID(hit->GetTrackID());
    		  Digi->SetParentID(hit->GetParentID());
    		  Digi->SetSourceEnergy(hit->GetSourceEnergy());
    		  Digi->SetSourcePDG(hit->GetSourcePDG());
    		  Digi->SetNCrystalConv( hit->GetNCrystalConv() );

    		  //-------------------------------------------------

    		    if (hit->GetComptonVolumeName().empty()) {
    		      Digi->SetComptonVolumeName( "NULL" );
    		      Digi->SetSourceID( -1 );
    		    }

    		    if (hit->GetRayleighVolumeName().empty()) {
    		      Digi->SetRayleighVolumeName( "NULL" );
    		      Digi->SetSourceID( -1 );
    		    }

    		/* //  if (nVerboseLevel>1)
    		        	G4cout << "[GateDigitizerInitializationModule::Digitize]: \n"
    		  	       << "\tprocessed " << *hit << Gateendl
    		  	       << "\tcreated new Digi:\n"
    		  	       << Digi << Gateendl
    		  	       << *Digi << Gateendl;
*/
    		  m_outputDigiCollection->insert(Digi);

    	  }



		}
   }
  StoreDigiCollection(m_outputDigiCollection);

}

void GateDigitizerInitializationModule::DescribeMyself(size_t )
{
  ;
}







