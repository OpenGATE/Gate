/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

// OK GND 2022
/*!
  \class  GateDiscretizer (by marc.granado@universite-paris-saclay.fr)
  \brief   for discretizing the position within a monolithic crystal

    - For each volume the local position of the interactions within the crystal are  discretized.
	the X,Y,Z vector is translated into the IDs of virtual divisions within the crystal
	occupying the levels of SubmoduleID, CrystalID and LayerID.

*/



#include "GateDiscretizerModule.hh"
#include "GateDiscretizerModuleMessenger.hh"
#include "GateDigi.hh"

#include "GateSpatialResolution.hh"
#include "GateSpatialResolutionMessenger.hh"



#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"

#include <iostream>
#include <cmath>
#include <limits>






GateDiscretizerModule::GateDiscretizerModule(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),

   m_nameAxis("XYZ"),

   m_resolution(0),
   m_resolutionX(0),
   m_resolutionY(0),
   m_resolutionZ(0),

   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateDiscretizerModuleMessenger(this);
}




GateDiscretizerModule::~GateDiscretizerModule()
{
  delete m_Messenger;

}






void GateDiscretizerModule::Digitize()
{



	GateSpatialResolution* digi_SpatialResolution;
	digi_SpatialResolution =  dynamic_cast<GateSpatialResolution*>(m_digitizer->FindDigitizerModule("digitizerMgr/"
	    		  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  +m_digitizer->GetSD()->GetName()
																						  +"/SinglesDigitizer/"
																						  + m_digitizer->GetName()
																						  + "/SpatialResolution"));



	G4double resolutionX;
	G4double resolutionY;
	G4double resolutionZ;










	if (digi_SpatialResolution->GetFWHM())
	{
		if(m_nameAxis.find('X') != std::string::npos)
			{resolutionX = digi_SpatialResolution->GetFWHM();}

		if(m_nameAxis.find('Y') != std::string::npos)
			{resolutionY = digi_SpatialResolution->GetFWHM();}

		if(m_nameAxis.find('Z') != std::string::npos)
			{resolutionZ = digi_SpatialResolution->GetFWHM();}
	}


	else if(digi_SpatialResolution->GetFWHMx() || digi_SpatialResolution->GetFWHMy()|| digi_SpatialResolution->GetFWHMz()){
		if(digi_SpatialResolution->GetFWHMx() && m_nameAxis.find('X') != std::string::npos)
				{resolutionX = digi_SpatialResolution->GetFWHMx();}

		if(digi_SpatialResolution->GetFWHMy() && m_nameAxis.find('Y') != std::string::npos)
				{resolutionY = digi_SpatialResolution->GetFWHMy();}

		if(digi_SpatialResolution->GetFWHMz() && m_nameAxis.find('Z') != std::string::npos)
				{resolutionZ = digi_SpatialResolution->GetFWHMz();}
	}

	else if (m_nameAxis == "XYZ" && (m_resolution==0) || (m_resolutionX==0 || m_resolutionY==0 ||m_resolutionZ==0))
	{

		GateError("***ERROR*** Discretization in XYZ has been selected but at least one axis has resolution = 0. Please ensure that all axis have a non zero value of resolution.");
	}


	if (m_resolution != 0)
	{
		resolutionX = m_resolution;
		resolutionY = m_resolution;
		resolutionZ = m_resolution;
	}

	else
	{
		resolutionX = m_resolutionX;
		resolutionY = m_resolutionY;
		resolutionZ = m_resolutionZ;
	}



	//TODO Create the check that tells you that
	GateVSystem* m_system =  ((GateSinglesDigitizer*)this->GetDigitizer())->GetSystem();

	if (m_system==NULL) G4Exception( "GateSpatialResolution::Digitize", "Digitize", FatalException,
				 "Failed to get the system corresponding to that digitizer. Abort.\n");

	if (!m_system->CheckIfEnoughLevelsAreDefined())
	{
		 GateError( " *** ERROR*** GateSpatialResolution::Digitize. Not all defined geometry levels has their mother levels defined."
				 "(Ex.: for cylindricalPET, the levels are: rsector, module, submodule, crystal). If you have defined submodule, you have to have resector and module defined as well."
				 "Please, add them to your geometry macro in /gate/systems/cylindricalPET/XXX/attach    YYY. Abort.\n");
	}

	m_systemDepth = m_system->GetTreeDepth();
	//TODO I need to create another parameter asking if spatial resolution is provided by hand.
	//I need to get system, system depth. Check that the size for the last 4 solids is the same (Monolithic crystal)
	//Need to access spatial resolution




	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	//What about these lines? Why are they not in SpatialResolution? Apparetnly
	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;



  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];
		  // ***** the following part of the code to adapt
		  /// *** This part is from ProcessPulseList

		  ////// ** This part is from ProcessOnePulse
		     for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
		     {
		    	 if ( (*iter)->GetVolumeID()   == inputDigi->GetVolumeID() )
		    	 {
		    		continue;
		    		if(m_nameAxis=="XYZ"){continue;}
		    		/* if(m_parameter=="numAxis"){
		                 DummyMethod1(inputDigi);
		    		 }
		    		 else{
		    			 DummyMethod2( inputDigi );
		    		 }
					*/
		      if (nVerboseLevel>1)
		      {
		    	 G4cout << "Merged previous pulse for volume " << inputDigi->GetVolumeID()
		 		 << " with new pulse of energy " << G4BestUnit(inputDigi->GetEnergy(),"Energy") <<".\n"
		 		 << "Resulting pulse is: \n"
		 		 << **iter << Gateendl << Gateendl ;
		      }
		 	  break;
		       }
		     }

		     if ( iter == OutputDigiCollectionVector->end() )
		     {
		       m_outputDigi = new GateDigi(*inputDigi);
		       m_outputDigi->SetEnergyIniTrack(-1);
		       m_outputDigi->SetEnergyFin(-1);
		       if (nVerboseLevel>1)
		 	  G4cout << "Created new pulse for volume " << inputDigi->GetVolumeID() << ".\n"
		 		 << "Resulting pulse is: \n"
		 		 << *m_outputDigi << Gateendl << Gateendl ;
		      /// !!!!!! The following line should be kept !!!! -> inserts the outputdigi to collection
		       m_OutputDigiCollection->insert(m_outputDigi);

		     }
		 ////// ** End of the part from ProcessOnePulse


		if (nVerboseLevel==1) {
			G4cout << "[GateDiscretizerModule::Digitize]: returning output pulse-list with " << OutputDigiCollectionVector->size() << " entries\n";
			for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
				G4cout << **iter << Gateendl;
			G4cout << Gateendl;
		}
	/// *** End of the part from ProcessPulseList
	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateDiscretizerModule::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}









void GateDiscretizerModule::SetNameAxis(const G4String &param)
{
	m_nameAxis=param;
}


///////////////////////////////////////////
////////////// Methods of DM //////////////
///////////////////////////////////////////




G4double GateDiscretizerModule::calculatePitch(G4double crystal_size, G4double spatial_resolution) {
         // Target pitch size
         double target_pitch = spatial_resolution / 2.0;
         double best_pitch = 0;
         double min_diff = std::numeric_limits<double>::max();

         // Iterate to find the best pitch size
         for (int num_pitches = 1; num_pitches <= crystal_size; ++num_pitches) {
             double pitch = crystal_size / num_pitches;
             if (std::fabs(pitch - target_pitch) < min_diff && std::fmod(crystal_size, pitch) == 0) {
                 min_diff = std::fabs(pitch - target_pitch);
                 best_pitch = pitch;
             }
         }

         return best_pitch;
     }




void GateDiscretizerModule::SetVirtualIDs( int nBinsX, int nBinsY,int nBinsZ, G4ThreeVector& pos ){
     int crystalID;
     int subModuleID;
     int layerID;

     int binX,binY,binZ;

	 binX = (pos.X/nBinsX+nBinsX/2);
	 binY = (pos.Y/nBinsY+nBinsY/2);
	 binZ = (pos.Z/nBinsY+nBinsY/2);

	 //Change the OutputVolumeID at depths 3,4,5
	 //(SubmoduleID=binX, crystalID = binY and layerID = binZ)
	 m_outputDigi->SetOutputVolumeID(3,binX);
	 m_outputDigi->SetOutputVolumeID(4,binY);
	 m_outputDigi->SetOutputVolumeID(5,binZ);
 }




/*
void GateDiscretizerModule::DummyMethod1(GateDigi *right)
{
	//to copy all variables that are not changed
	m_outputDigi=right;


}

void GateDiscretizerModule::DummyMethod2(GateDigi *right)
{
	//to copy all variables that are not changed
	m_outputDigi=right;


}

*/
void GateDiscretizerModule::DescribeMyself(size_t indent )
{
	if(m_resolution)
	  G4cout << GateTools::Indent(indent) << "Spatial resolution : " << m_resolution  << Gateendl;
	else
	  G4cout << GateTools::Indent(indent) << "Spatial resolution : " << m_resolutionX <<" "<< m_resolutionY<< " "<<m_resolutionZ<< Gateendl;
;
}















