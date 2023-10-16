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

	m_outputDigiCollection = new GateDigiCollection (GetName(),  m_digitizer->GetOutputName() ); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();
	G4String HCname=m_digitizer->m_SD->GetName()+"Collection" ;



	if (m_FirstEvent)
	{
		m_HCID= DigiMan->GetHitsCollectionID(HCname);
		m_FirstEvent=false;
	}


	GateHitsCollection* inHC = (GateHitsCollection*) (DigiMan->GetHitsCollection(m_HCID));// DigiMan->GetHitsCollectionID(HCname)));


	if (inHC)
    {
      G4int n_hit = inHC->entries();

	 for (G4int i=0;i<n_hit;i++)
	{
    	 if((*inHC)[i]->GetEdep() !=0 )
    	  {
    		  GateDigi* Digi = new GateDigi();
    		  Digi->SetMother( (*inHC)[i] );
    		  Digi->SetRunID( (*inHC)[i]->GetRunID() );
    		  Digi->SetEventID( (*inHC)[i]->GetEventID() );
    		  Digi->SetTrackID( (*inHC)[i]->GetTrackID() );
    		  Digi->SetSourceID( (*inHC)[i]->GetSourceID() );
    		  Digi->SetSourcePosition( (*inHC)[i]->GetSourcePosition() );
    		  Digi->SetTime( (*inHC)[i]->GetTime() );
    		  Digi->SetEnergy( (*inHC)[i]->GetEdep() );
    		  Digi->SetMaxEnergy( (*inHC)[i]->GetEdep() );
    		  Digi->SetLocalPos( (*inHC)[i]->GetLocalPos() );
    		  Digi->SetGlobalPos( (*inHC)[i]->GetGlobalPos() );
    		  Digi->SetPDGEncoding( (*inHC)[i]->GetPDGEncoding() );
    		  Digi->SetOutputVolumeID( (*inHC)[i]->GetOutputVolumeID() );
    		  Digi->SetNPhantomCompton( (*inHC)[i]->GetNPhantomCompton() );
    		  Digi->SetNCrystalCompton( (*inHC)[i]->GetNCrystalCompton() );
    		  Digi->SetNPhantomRayleigh( (*inHC)[i]->GetNPhantomRayleigh() );
    		  Digi->SetNCrystalRayleigh( (*inHC)[i]->GetNCrystalRayleigh() );
    		  Digi->SetComptonVolumeName( (*inHC)[i]->GetComptonVolumeName() );
    		  Digi->SetRayleighVolumeName( (*inHC)[i]->GetRayleighVolumeName() );
    		  Digi->SetVolumeID( (*inHC)[i]->GetVolumeID() );
    		  Digi->SetSystemID( (*inHC)[i]->GetSystemID() );
    		  Digi->SetScannerPos( (*inHC)[i]->GetScannerPos() );
    		  Digi->SetScannerRotAngle( (*inHC)[i]->GetScannerRotAngle() );
    		  #ifdef GATE_USE_OPTICAL
    		    Digi->SetOptical( (*inHC)[i]->GetPDGEncoding() == -22);
    		  #endif
    		  Digi->SetNSeptal( (*inHC)[i]->GetNSeptal() );  // HDS : septal penetration

    		  // AE : Added for IdealComptonPhot adder which take into account several Comptons in the same volume
    		  Digi->SetPostStepProcess((*inHC)[i]->GetPostStepProcess());
    		  Digi->SetEnergyIniTrack((*inHC)[i]->GetEnergyIniTrack());
    		  Digi->SetEnergyFin((*inHC)[i]->GetEnergyFin());
    		  Digi->SetProcessCreator((*inHC)[i]->GetProcess());
    		  Digi->SetTrackID((*inHC)[i]->GetTrackID());
    		  Digi->SetParentID((*inHC)[i]->GetParentID());
    		  Digi->SetSourceEnergy((*inHC)[i]->GetSourceEnergy());
    		  Digi->SetSourcePDG((*inHC)[i]->GetSourcePDG());
    		  Digi->SetNCrystalConv( (*inHC)[i]->GetNCrystalConv() );

    		  //-------------------------------------------------

    		    if ((*inHC)[i]->GetComptonVolumeName().empty()) {
    		      Digi->SetComptonVolumeName( "NULL" );
    		      Digi->SetSourceID( -1 );
    		    }

    		    if ((*inHC)[i]->GetRayleighVolumeName().empty()) {
    		      Digi->SetRayleighVolumeName( "NULL" );
    		      Digi->SetSourceID( -1 );
    		    }

    		/* //  if (nVerboseLevel>1)
    		        	G4cout << "[GateDigitizerInitializationModule::Digitize]: \n"
    		  	       << "\tprocessed " << *(*inHC)[i] << Gateendl
    		  	       << "\tcreated new Digi:\n"
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







