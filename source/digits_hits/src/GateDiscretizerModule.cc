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

#include "G4Box.hh"
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


//From here

	GateSpatialResolution* digi_SpatialResolution;
	digi_SpatialResolution =  dynamic_cast<GateSpatialResolution*>(m_digitizer->FindDigitizerModule("digitizerMgr/"
	    		  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  +m_digitizer->GetSD()->GetName()
																						  +"/SinglesDigitizer/"
																						  + m_digitizer->GetName()
																						  + "/spatialResolution"));



	G4double resolutionX;
	G4double resolutionY;
	G4double resolutionZ;

	G4int nBinsX;
	G4int nBinsY;
	G4int nBinsZ;

	G4double pitchX;
	G4double pitchY;
	G4double pitchZ;



	//TODO bulletproof this!

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



	G4double xLength,yLength,zLength;



  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];
		  // ***** the following part of the code to adapt
		  /// *** This part is from ProcessPulseList
		  if (inputDigi->GetVolumeID().size()){
		  G4Box* box = dynamic_cast<G4Box*>(inputDigi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid());
	        xLength = 2*box->GetXHalfLength();
	        yLength = 2*box->GetYHalfLength();
	        zLength = 2*box->GetZHalfLength();
		  }
		  else std::cout<<"Well that's another problem here..."<< std::endl;






		      if (nVerboseLevel>1)
		      {
		    	 G4cout << "Merged previous pulse for volume " << inputDigi->GetVolumeID()
		 		 << " with new pulse of energy " << G4BestUnit(inputDigi->GetEnergy(),"Energy") <<".\n"
		 		 << "Resulting pulse is: \n"
		 		 << **iter << Gateendl << Gateendl ;
		      }


		       m_outputDigi = new GateDigi(*inputDigi);
		       m_outputDigi->SetEnergyIniTrack(-1);
		       m_outputDigi->SetEnergyFin(-1);




		       if(m_nameAxis=="XYZ"){

		       			    		pitchX = calculatePitch(xLength,resolutionX);
		       			    		nBinsX = int(xLength/pitchX);
		       			    		pitchY = calculatePitch(yLength,resolutionY);
		       			    		nBinsY = int(yLength/pitchY);
		       			    		pitchZ = calculatePitch(zLength,resolutionZ);
		       			    		nBinsZ = int(zLength/pitchZ);


		       			    		G4ThreeVector localPos = inputDigi->GetLocalPos();
		       			    		//std::cout<<"Local position = " << localPos <<" nBinsX "<<nBinsX<<" nBinsY "<<nBinsY<<" nBinsZ "<<nBinsZ<<std::endl;
		       			    		SetVirtualIDs(nBinsX,nBinsY,nBinsZ,pitchX,pitchY,pitchZ,localPos);

		       		    			}

		       std::cout<<"m_outputdigi "<<m_outputDigi->GetOutputVolumeID()<<std::endl;


		       if (nVerboseLevel>1)
		 	  G4cout << "Created new pulse for volume " << inputDigi->GetVolumeID() << ".\n"
		 		 << "Resulting pulse is: \n"
		 		 << *m_outputDigi << Gateendl << Gateendl ;
		      /// !!!!!! The following line should be kept !!!! -> inserts the outputdigi to collection
		       m_OutputDigiCollection->insert(m_outputDigi);


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



/*
G4double GateDiscretizerModule::calculatePitch(G4double crystal_size, G4double spatial_resolution) {


	// Target pitch size
         double target_pitch = spatial_resolution / 2.0;
         std::cout<<"Spatial Resolution: "<<spatial_resolution<<std::endl;
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

         std::cout<<"Best pitch is: "<<best_pitch<<" weird since the condition is to be smaller than: "<< min_diff<<std::endl;

         return best_pitch;
     }
*/

//TODO: This function needs much optimising
G4double GateDiscretizerModule::calculatePitch(G4double crystal_size, G4double spatial_resolution) {
    // Target pitch size
    double target_pitch = spatial_resolution / 2.0;
    double best_pitch = 0;
    double min_diff = 999;
    int num_pitches = 1;

    while (true) {
        double pitch = crystal_size / num_pitches;

        // Check if pitch meets both conditions
        if (pitch <= target_pitch && std::fmod(crystal_size, pitch) == 0) {
            double diff = std::fabs(pitch - target_pitch);
            if (diff <= min_diff) {
                min_diff = diff;
                best_pitch = pitch;
            }
        }

        // Break if pitch is smaller than a threshold, to prevent infinite loop
        if ((best_pitch <= target_pitch)&&(best_pitch!=0)) {
            break;
        }
        else if (num_pitches>1000){
        	std::cout<<"ERROR! No pitch found in 1000 iterations!"<<std::endl;
        	break;
        }

        ++num_pitches;
    }


    return best_pitch;
}


void GateDiscretizerModule::SetVirtualIDs( int nBinsX, int nBinsY,int nBinsZ,double pitchX,double pitchY,double pitchZ, G4ThreeVector& pos ){


     int binX,binY,binZ;



	 binX = static_cast<int>(pos.getX()/pitchX)+static_cast<int>(nBinsX/2);
	 binY = static_cast<int>(pos.getY()/pitchY)+static_cast<int>(nBinsY/2);
	 binZ = static_cast<int>(pos.getZ()/pitchZ)+static_cast<int>(nBinsY/2);


	 //Change the OutputVolumeID at depths 3,4,5
	 //(SubmoduleID=binX, crystalID = binY and layerID = binZ)
	 m_outputDigi->SetOutputVolumeID(binX,3);
	 m_outputDigi->SetOutputVolumeID(binY,4);
	 m_outputDigi->SetOutputVolumeID(binZ,5);


 }





void GateDiscretizerModule::SetVirtualID( int nBins, double pitch, G4double pos , int depth){



     int bin;


	 bin = static_cast<int>(pos/pitch)+static_cast<int>(nBins/2);


	 m_outputDigi->SetOutputVolumeID(bin,depth);

 }




void GateDiscretizerModule::DescribeMyself(size_t indent )
{
	if(m_resolution)
	  G4cout << GateTools::Indent(indent) << "Spatial resolution : " << m_resolution  << Gateendl;
	else
	  G4cout << GateTools::Indent(indent) << "Spatial resolution : " << m_resolutionX <<" "<< m_resolutionY<< " "<<m_resolutionZ<< Gateendl;
;
}















