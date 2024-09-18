/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

// OK GND 2022
/*!
  \class  GateVirtualSegmentationSD (by marc.granado@universite-paris-saclay.fr)
  \brief   for discretizing the position through a virtual segmentation of a monolithic crystal

    - For each volume the local position of the interactions within the crystal are virtually segmented.
	the X,Y,Z vector is translated into the IDs of virtual divisions within the crystal
	occupying the levels of SubmoduleID, CrystalID and LayerID.

*/



#include "GateVirtualSegmentationSD.hh"
#include "GateVirtualSegmentationSDMessenger.hh"
#include "GateDigi.hh"

#include "GateSpatialResolution.hh"
#include "GateSpatialResolutionMessenger.hh"

#include "GateBoxComponent.hh"

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
#include <cstdlib>





GateVirtualSegmentationSD::GateVirtualSegmentationSD(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),

   m_IsFirstEntry(1),
   m_nameAxis("XYZ"),
   useMacroGenerator(0),

   m_pitch(0),
   m_pitchX(0),
   m_pitchY(0),
   m_pitchZ(0),

   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer),

	pitchX(0),
	pitchY(0),
	pitchZ(0),

	nBinsX(0),
	nBinsY(0),
	nBinsZ(0),

	depthX(5),
	depthY(4),
	depthZ(3),

	xLength(0),
	yLength(0),
	zLength(0)


{
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateVirtualSegmentationSDMessenger(this);
}




GateVirtualSegmentationSD::~GateVirtualSegmentationSD()
{
  delete m_Messenger;

}






void GateVirtualSegmentationSD::Digitize()
{

	GateVSystem* m_system =  ((GateSinglesDigitizer*)this->GetDigitizer())->GetSystem();

	if (m_system==NULL) G4Exception( "GateVirtualSegmentationSD::Digitize", "Digitize", FatalException,
				 "Failed to get the system corresponding to that digitizer. Abort.\n");

	if (!m_system->CheckIfEnoughLevelsAreDefined())
	{
		 GateError( " *** ERROR*** GateVirtualSegmentationSD::Digitize. Not all defined geometry levels has their mother levels defined."
				 "(Ex.: for cylindricalPET, the levels are: rsector, module, submodule, crystal). If you have defined submodule, you have to have resector and module defined as well."
				 "Please, add them to your geometry macro in /gate/systems/cylindricalPET/XXX/attach    YYY. Abort.\n");
	}



	m_systemDepth = m_system->GetTreeDepth();

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

		  if (inputDigi->GetVolumeID().size()){
		  G4Box* box = dynamic_cast<G4Box*>(inputDigi->GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid());
	        xLength = 2*box->GetXHalfLength();
	        yLength = 2*box->GetYHalfLength();
	        zLength = 2*box->GetZHalfLength();
		  }
		  else GateError( " *** ERROR*** No volumeID in inputDigi!");




			if (m_IsFirstEntry){

				if(pitchX > xLength){ pitchX=xLength;
				GateWarning("The size provided for pitchX is bigger than the crystal syze! The length of the crystal has been selected as the new pitchX");}
				if(pitchY > yLength){ pitchY=yLength;
				GateWarning("The size provided for pitchY is bigger than the crystal syze! The length of the crystal has been selected as the new pitchY");}
				if(pitchZ > zLength){ pitchZ=zLength;
				GateWarning("The size provided for pitchZ is bigger than the crystal syze! The length of the crystal has been selected as the new pitchX");}

				SetParameters();



				if (useMacroGenerator){
					/*
					std::string path_to_script = "/home/granado/GATE_projects/MonoCrystals/tests/mac/";
					std::string path_to_macros = "/home/granado/GATE_projects/MonoCrystals/tests/mac/";
				//Once this is implemented as a GateTool it will be:
				//std::string command = "macro_converter";
					std::string command = "python3 "+path_to_script+"macro_converter.py";
					command +=" -d "+path_to_macros+"digitizer.mac";
					command +=" -g "+path_to_macros+"geometry_pseudo-crystal.mac";

					if (pitchX) command +=" -x "+std::to_string(pitchX);
					if (pitchY) command +=" -y "+std::to_string(pitchY);
					if (pitchZ) command +=" -z "+std::to_string(pitchZ);

					if (nBinsX) command +=" --binsX "+std::to_string(nBinsX);
					if (nBinsY) command +=" --binsY "+std::to_string(nBinsY);
					if (nBinsZ) command +=" --binsZ "+std::to_string(nBinsZ);

					int result = system(command.c_str());
					if (result ==0) std::cout<<"The macro converter script has been executed corectly"<<std::endl;
					else std::cout<<"There was an error in the execution of the macro converter script"<<std::endl;
				*/
					std::cout<<"Sorry, the macro converter script has not yet been implemented"<<std::endl;
				}

				m_IsFirstEntry=0;
			}



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


		       //Accessing the local position of the hit to define and set the new VirtualID

		       G4ThreeVector localPos = inputDigi->GetLocalPos();

		       if(m_nameAxis.find('X') != std::string::npos){
		    	   SetVirtualID(nBinsX,pitchX,localPos.getX(),depthX);
		       }
		       if(m_nameAxis.find('Y') != std::string::npos){
		    	   SetVirtualID(nBinsY,pitchY,localPos.getY(),depthY);
		       }
		       if(m_nameAxis.find('Z') != std::string::npos){
		    	   SetVirtualID(nBinsZ,pitchZ,localPos.getZ(),depthZ);
		       }







		       if (nVerboseLevel>1)
		 	  G4cout << "Created new pulse for volume " << inputDigi->GetVolumeID() << ".\n"
		 		 << "Resulting pulse is: \n"
		 		 << *m_outputDigi << Gateendl << Gateendl ;
		      /// !!!!!! The following line should be kept !!!! -> inserts the outputdigi to collection
		       m_OutputDigiCollection->insert(m_outputDigi);


		 ////// ** End of the part from ProcessOnePulse


		if (nVerboseLevel==1) {
			G4cout << "[GateVirtualSegmentationSD::Digitize]: returning output pulse-list with " << OutputDigiCollectionVector->size() << " entries\n";
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
  	  	G4cout << "[GateVirtualSegmentationSD::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);


}









void GateVirtualSegmentationSD::SetNameAxis(const G4String &param)
{
	m_nameAxis=param;
}


///////////////////////////////////////////
////////////// Methods of DM //////////////
///////////////////////////////////////////


//TODO: This function could be further optimised (but only used once)

G4double GateVirtualSegmentationSD::calculatePitch(G4double crystal_size, G4double target_pitch) {
    // Target pitch size

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
        	std::cout<<"ERROR! No pitch found in 1000 iterations! Size = "<<crystal_size<<" and desired pitch = "<<pitch<<std::endl;
        	break;
        }

        ++num_pitches;
    }


    return best_pitch;
}




void GateVirtualSegmentationSD::SetVirtualID( int nBins, double pitch, G4double pos , int depth){

     int bin;


	 bin = std::floor(pos/pitch+nBins/2.);
	 m_outputDigi->SetOutputVolumeID(bin,depth);

 }




void GateVirtualSegmentationSD::DescribeMyself(size_t indent )
{
	if(m_pitch)
	  G4cout << GateTools::Indent(indent) << "Spatial pitch : " << m_pitch  << Gateendl;
	else
	  G4cout << GateTools::Indent(indent) << "Spatial pitch : " << m_pitchX <<" "<< m_pitchY<< " "<<m_pitchZ<< Gateendl;
;
}







void GateVirtualSegmentationSD::SetParameters()

{


	//Setting the global parameters



	//Obtaining the spatial resolution digitizer to use the its values for the definition of the pitch in case no pitch has been provided by the user
	GateSpatialResolution* digi_SpatialResolution;
		digi_SpatialResolution =  dynamic_cast<GateSpatialResolution*>(m_digitizer->FindDigitizerModule("digitizerMgr/"
		    		  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  +m_digitizer->GetSD()->GetName()
																							  +"/SinglesDigitizer/"
																							  + m_digitizer->GetName()

																							  + "/spatialResolution"));


	//Setting the pitch size provided by the user

	if(m_pitch){
			if(m_nameAxis.find('X') != std::string::npos)
				{pitchX = m_pitch;}

			if(m_nameAxis.find('Y') != std::string::npos)
				{pitchY = m_pitch;}

			if(m_nameAxis.find('Z') != std::string::npos)
				{pitchZ = m_pitch;}


			if	(m_pitchX!=0 || m_pitchY!=0 ||m_pitchZ!=0)
				{GateWarning("The values provided for target pitch are ambiguous. Only the value of 'pitch' was taken, any value provided for 'pitchX', 'pitchY' and 'pitchZ' was ignored");}
		}

		else if	(m_pitchX!=0 || m_pitchY!=0 ||m_pitchZ!=0){

			if(m_pitchX && m_nameAxis.find('X') != std::string::npos)
					{pitchX = m_pitchX;}

			if(m_pitchY && m_nameAxis.find('Y') != std::string::npos)
					{pitchY = m_pitchY;}

			if(m_pitchZ && m_nameAxis.find('Z') != std::string::npos)
					{pitchZ = m_pitchZ;}

		}

		else if (digi_SpatialResolution){
			if (digi_SpatialResolution->GetFWHM())
			{

				if(m_nameAxis.find('X') != std::string::npos)
					{pitchX = 0.5*digi_SpatialResolution->GetFWHM();}

				if(m_nameAxis.find('Y') != std::string::npos)
					{pitchY = 0.5*digi_SpatialResolution->GetFWHM();}

				if(m_nameAxis.find('Z') != std::string::npos)
					{pitchZ = 0.5*digi_SpatialResolution->GetFWHM();}
			}


			else if(digi_SpatialResolution->GetFWHMx() || digi_SpatialResolution->GetFWHMy()|| digi_SpatialResolution->GetFWHMz()){



				if(digi_SpatialResolution->GetFWHMxdistrib()||digi_SpatialResolution->GetFWHMydistrib()||digi_SpatialResolution->GetFWHMxydistrib2D())
				 {
				 GateError("***ERROR*** No value of the target pitch has been provided and no value can be obtained from the spatial resolution distribution. /n Please provide a value for the pitch that is at least half of the minimum value of the distribution. ");
				 }


	

				if(digi_SpatialResolution->GetFWHMx() && m_nameAxis.find('X') != std::string::npos)
					{pitchX = 0.5*digi_SpatialResolution->GetFWHMx();}

				if(digi_SpatialResolution->GetFWHMy() && m_nameAxis.find('Y') != std::string::npos)
					{pitchY = 0.5*digi_SpatialResolution->GetFWHMy();}

				if(digi_SpatialResolution->GetFWHMz() && m_nameAxis.find('Z') != std::string::npos)
					{pitchZ = 0.5*digi_SpatialResolution->GetFWHMz();}
			}
		}

		if(pitchX == 0 && m_nameAxis.find('X') != std::string::npos)
				{GateError("***ERROR*** Virtual setmentation in X axis has been selected but no value for FWHM was provided.");
		}
		else if (pitchX!=0){

				std::cout<<"Calculating pitchX with = "<<pitchX<<std::endl;
				pitchX = calculatePitch(xLength,pitchX);
				std::cout<<"Resulting pitchX = "<<pitchX<<std::endl;
				nBinsX = int(xLength/pitchX);
		}

		if(pitchY == 0 && m_nameAxis.find('Y') != std::string::npos)
				{GateError("***ERROR*** Virtual setmentation in Y axis has been selected but no value for FWHM was provided.");
		}
		else if (pitchY!=0){
			   std::cout<<"Calculating pitchY with = "<<pitchY<<std::endl;
	    	   pitchY = calculatePitch(yLength,pitchY);
	    	   std::cout<<"Resulting pitchY = "<<pitchY<<std::endl;
	    	   nBinsY = int(yLength/pitchY);
		}



		if(pitchZ == 0 && m_nameAxis.find('Z') != std::string::npos)
				{GateError("***ERROR*** Virtual setmentation in Z axis has been selected but no value for FWHM was provided.");
		}
		else if (pitchZ!=0){

			   std::cout<<"Calculating pitchZ with = "<<pitchZ<<std::endl;
	    	   pitchZ = calculatePitch(zLength,pitchZ);
	    	   std::cout<<"Resulting pitchZ = "<<pitchZ<<std::endl;
	    	   nBinsZ = int(zLength/pitchZ);
		}




		//setting the levels of depth for each variable from bottom to top in order XYZ

	       if(m_nameAxis.find('X') != std::string::npos){
    		   depthX=depthX;
	    	   if(m_nameAxis.find('Y') != std::string::npos){
	    		   depthY=depthY;
				if(m_nameAxis.find('Z') != std::string::npos){
		    		   depthZ=depthZ;
	    		   }
	    	   }
	    	   else if(m_nameAxis.find('Z') != std::string::npos){

	    		   depthZ+=1;
	    	   	   }
	       }
	       else if(m_nameAxis.find('Y') != std::string::npos){

	    	   depthY+=1;

	    	   if(m_nameAxis.find('Z') != std::string::npos){

	    		   depthZ+=1;
	    	   }
	       }
	       else if(m_nameAxis.find('Z') != std::string::npos){
	    	   	 	 depthZ+=2;
	       }

	       else GateError("Not entering any of the depths!");





	       //Testing there are no repeaters in the depth levels needed for the virtual ID's.

	       GateVSystem* m_system =  ((GateSinglesDigitizer*)this->GetDigitizer())->GetSystem();


	       GateSystemComponent* m_submoduleComponent = m_system->FindComponent("submodule");
	       GateSystemComponent* m_crystalComponent = m_system->FindComponent("crystal");
	       GateSystemComponent* m_layer0Component= m_system->FindComponent("layer0");
	       std::cout<<"System depth is: "<<m_systemDepth<<std::endl;
	       G4int repeaterNumber = 0;



	       repeaterNumber = m_layer0Component-> GetVolumeNumber();
      	   if (repeaterNumber > 1)
	       	        GateError( " *** ERROR*** GateVirtualSegmentationSD::Digitize. Crystal level needs to be empty (no repeaters) to use VirtualSegmentationSD with XYZ segmentation");



	       	if (m_nameAxis.size()>=2)
	       	{
	       	   repeaterNumber = m_crystalComponent-> GetVolumeNumber();
	       	   if (repeaterNumber > 1)
	       	        GateError( " *** ERROR*** GateVirtualSegmentationSD::Digitize. Crystal level needs to be empty (no repeaters) to use VirtualSegmentationSD with XYZ segmentation");

	       	}




	       if (m_nameAxis.size()==3)
	       {
        	   repeaterNumber = m_submoduleComponent-> GetVolumeNumber();
	           if (repeaterNumber >1)
	        	   	   GateError( " *** ERROR*** GateVirtualSegmentationSD::Digitize. Submodule level needs to be empty (no repeaters) to use VirtualSegmentationSD with XYZ segmentation");

	       	}

}






