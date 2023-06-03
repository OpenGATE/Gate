
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateReadout

 S. Stute - June 2014: complete redesign of the readout module and add a new policy to emulate PMT.
    - Fix bug in choosing the maximum energy pulse.We now have some temporary lists of variables to deal
      with the output pulses. These output pulses are only created at the end of the method. In previous
      versions, the output pulse was accumulating energy as long as input pulses were merged together, but
      the problem is that the comparison of energy was done between the input pulse and this output pulse
      with increasing energy. So with more than 2 pulses to be merged together, the behaviour was undefined.
    - Move all the processing into the upper method ProcessPulseList instead of using the mother version
      working into the ProcessOnePulse method. Thus the ProcessOnePulse in this class is not used anymore.
    - Create policy choice: now we can choose via the messenger between EnergyWinner and EnergyCentroid.
    - For the EnergyCentroid policy, the centroid position is computed using the crystal indices in each
      direction, doing the computation with floating point numbers, and then casting the result into
      integer indices. Using that method, we ensure the centroid position to be in a crystal (if we work
      with global position, we can fall between two crystals in the presence of gaps).
      The depth is ignored with this strategy; it is forced to be at one level above the 'crystal' level.
      If there a 'layer' level below the 'crystal' level, an energy winner strategy is adopted.

  O. Kochebina - April 2022: new messenger options are added and some minor bugs corrected

  O. Kochebina - September 2022: passing to GND
*/

#include "GateReadout.hh"
#include "GateReadoutMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"

#include "GateOutputVolumeID.hh"
#include "GateTools.hh"
#include "GateArrayComponent.hh"
#include "GateVSystem.hh"


GateReadout::GateReadout(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_depth(0),
   m_policy("TakeEnergyWinner"),
   m_IsFirstEntrance(1),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {

	// S. Stute: These variables are used for the energy centroid strategy
	m_nbCrystalsX  = -1;
	m_nbCrystalsY  = -1;
	m_nbCrystalsZ  = -1;
	m_nbCrystalsXY = -1;
	m_crystalDepth = -1;
	m_systemDepth  = -1;
	m_system = NULL;
	m_crystalComponent = NULL;
	m_IsForcedDepthCentroid = false;

	G4String colName = digitizer->GetOutputName();
	collectionName.push_back(colName);
	m_messenger = new GateReadoutMessenger(this);
}


GateReadout::~GateReadout()
{
  delete m_messenger;
}




void GateReadout::SetReadoutParameters()
{


	//checking the if depth or ReadoutVolumeName are defined and that only one is set.
	if(!m_volumeName.empty() && m_depth!=0)
		GateError("***ERROR*** You can choose Readout parameter either with /setDepth OR /setReadoutVolume!");

	 //////////////DEPTH SETTING/////////
	 //set the previously default value for compatibility of users macros
	if(m_volumeName.empty()  && m_depth==0)
		m_depth=1; //previously default value

	 //set m_depth according user defined volume name
	 if(!m_volumeName.empty()) //only for EnergyWinner
	 	 {
		 if(m_policy =="TakeEnergyCentroid"&& !m_IsForcedDepthCentroid)
			 GateError("***ERROR*** Please, remove /setDepth or /setReadoutVolume for TakeEnergyCentroid policy as this parameter is set automatically. "
					 "Use /forceReadoutVolumeForEnergyCentroid flag if you still want to set your depth/volume for Readout.\n");


		 	 GateVSystem* m_system = ((GateSinglesDigitizer*)this->GetDigitizer())->GetSystem();
		 	 if (m_system==NULL) G4Exception( "GateReadout::SetReadoutParameters", "SetReadoutParameters", FatalException,
	  	  	                                   "Failed to get the system corresponding to that digitizer. Abort.\n");

		 	 m_systemDepth = m_system->GetTreeDepth();

		 	 GateObjectStore* anInserterStore = GateObjectStore::GetInstance();
		 	 for (G4int i=1;i<m_systemDepth;i++)
		 	 {
		 		 GateSystemComponent* comp0= (m_system->MakeComponentListAtLevel(i))[0][0];
		 		 GateVVolume *creator = comp0->GetCreator();
		 		 GateVVolume* anInserter = anInserterStore->FindCreator(m_volumeName);

		 		 if(creator==anInserter)
		 			 m_depth=i;

	  	   }
	  	}


	 //////////////Resulting positioning SETTING/////////
	 //previously default conditions for compatibility of users macros
	/* if(m_resultingXY.empty() && m_resultingZ.empty() && m_policy =="TakeEnergyCentroid")
	 {
		 m_resultingXY="crystalCenter";
		 m_resultingZ="crystalCenter";
	 }
	 if (m_resultingXY.empty() && m_resultingZ.empty() && m_policy =="TakeEnergyWinner")
	 {
		 m_resultingXY="exactPostion";
		 m_resultingZ="exactPostion";
	 }
	 */


	if (m_policy=="TakeEnergyCentroid" && (!m_volumeName.empty()||m_depth) &&  !m_IsForcedDepthCentroid)
	 {
		GateWarning("WARNING! Commands /setDepth and /setReadoutVolume are ignored as Energy Centroid policy is used: "
				"the depth is forced to be at the level just above the crystal level, whatever the system used."
				"To force the depth, please, set the flag /forceReadoutVolumeForEnergyCentroid to true");
	 }

	if (m_policy=="TakeEnergyWinner" && m_IsForcedDepthCentroid)
		 {
		GateError("***ERROR*** Command /forceReadoutVolumeForEnergyCentroid can not be used for Winner policy. Abort.\n");
		 }


	if (m_policy=="TakeEnergyCentroid" )
		{

			// Find useful stuff for centroid based computation
			//m_policy = "TakeEnergyCentroid";
			// Get the system
			GateVSystem* m_system = ((GateSinglesDigitizer*)this->GetDigitizer())->GetSystem();
			if (m_system==NULL) G4Exception( "GateReadout::Digitize", "Digitize", FatalException,
					"Failed to get the system corresponding to that processor chain. Abort.\n");
			// Get the array component corresponding to the crystal level using the name 'crystal'
			GateArrayComponent* m_crystalComponent = m_system->FindArrayComponent("crystal");
			if (m_crystalComponent==NULL) G4Exception( "GateReadout::Digitize", "Digitize", FatalException,
												  "Failed to get the array component corresponding to the crystal. Abort.\n");

			if (!m_system->CheckIfAllLevelsAreDefined())
					{
						 GateError( " *** ERROR*** GateReadout::Digitize. Not all required geometry levels and sublevels for this system are defined. "
						       			  			  "(Ex.: for cylindricalPET, the required levels are: rsector, module, submodule, crystal). Please, add them to your geometry macro in /gate/systems/cylindricalPET/XXX/attach    YYY. Abort.\n");
					}


			// Get the number of crystals in each direction
			m_nbCrystalsZ  = m_crystalComponent->GetRepeatNumber(2);
			m_nbCrystalsY  = m_crystalComponent->GetRepeatNumber(1);
			m_nbCrystalsX  = m_crystalComponent->GetRepeatNumber(0);
			m_nbCrystalsXY = m_nbCrystalsX * m_nbCrystalsY;
			if (m_nbCrystalsX<1 || m_nbCrystalsY<1 || m_nbCrystalsZ<1)
				G4Exception( "GateReadout::Digitize", "Digitize", FatalException,
						"Crystal repeater numbers are wrong !\n");
			// Get tree depth of the system
			m_systemDepth = m_system->GetTreeDepth();
			//G4cout << "  Depth of the system: " << m_systemDepth << Gateendl;
			// Find the crystal depth in the system
			GateSystemComponent* this_component = m_system->GetBaseComponent();
			m_crystalDepth = 0;
			while (this_component!=m_crystalComponent && m_crystalDepth+1<m_systemDepth)
			{
				this_component = this_component->GetChildComponent(0);
				m_crystalDepth++;
			}
			if (this_component!=m_crystalComponent) G4Exception( "GateReadout::Digitize", "Digitize", FatalException,
																	"Failed to get the system depth corresponding to the crystal. Abort.\n");
			// Now force m_depth to be right above the crystal depth
			//m_depth = m_crystalDepth - 1;
		if (!m_IsForcedDepthCentroid)
			{
			m_depth = m_crystalDepth - 1;
			}

		}

	if (m_policy!="TakeEnergyCentroid" && m_policy!="TakeEnergyWinner")
		G4Exception( "GateReadout::SetPolicy", "SetPolicy", FatalException, "Unknown provided policy, please see the guidance. Abort.\n");

	//G4cout<<"Policy = "<< m_policy<< Gateendl;
	//G4cout<<"Depth =  "<< m_depth<<Gateendl;
	//G4cout<<"resultingXY = "<< m_resultingXY<<Gateendl;
	//G4cout<<"reulstingZ = "<< m_resultingZ<<Gateendl;

}

void GateReadout::Digitize()
{
  //G4cout<<" GateReadout::Digitize "<< m_IsFirstEntrance <<G4endl;

  G4String digitizerName = m_digitizer->m_digitizerName;
  G4String outputCollName = m_digitizer-> GetOutputName();
  //G4cout<<"outputCollName "<<outputCollName<<G4endl;
  m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

  G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();

  GateDigiCollection* IDC = 0;
  IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection( m_DCID ));

  GateDigi* inputDigi;


  if (!IDC)
	  return;

  G4int n_digi = IDC->entries();
  	  if (nVerboseLevel>1)
  		  G4cout << "[GateReadout::Digitize]: processing input list with " << n_digi << " entries\n";

  // S. Stute: these variables are used for the energy centroid policy
  G4double* final_time = NULL;
  G4double* final_crystal_posX = NULL;
  G4double* final_crystal_posY = NULL;
  G4double* final_crystal_posZ = NULL;
  G4double* final_energy = NULL;
  G4int* final_nb_digi = NULL;
  GateDigi** final_digi = NULL;

  G4int final_nb_out_digi = 0;


  if(m_IsFirstEntrance) //set parameters at the first iteration
  {
	  SetReadoutParameters();
	  m_IsFirstEntrance=0;
  }

  // S. Stute: these variables are used for the energy centroid policy
  final_time = NULL;
  final_crystal_posX = NULL;
  final_crystal_posY = NULL;
  final_crystal_posZ = NULL;
  final_energy = NULL;
  final_nb_digi = NULL;
  final_digi = NULL;

  if (m_policy=="TakeEnergyCentroid")
  {
	  final_time         = (G4double*)calloc(n_digi,sizeof(G4double));
	  final_crystal_posX = (G4double*)calloc(n_digi,sizeof(G4double));
	  final_crystal_posY = (G4double*)calloc(n_digi,sizeof(G4double));
	  final_crystal_posZ = (G4double*)calloc(n_digi,sizeof(G4double));
	  final_nb_digi    = (G4int*)calloc(n_digi,sizeof(G4int));
  }
  // S. Stute: we need energy to sum up correctly for all output digi and affect only at the end.
  //           In previous versions, even for take Winner, the energy was affected online so the
  //           final digi was not the winner in all cases.
  final_energy = (G4double*)calloc(n_digi,sizeof(G4double));
  final_digi = (GateDigi**)calloc(n_digi,sizeof(GateDigi*));
  final_nb_out_digi = 0;


  // Start loop on input pulses
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];
		  //G4cout<<"inHC "  << IDC->GetName ()<<" "<<  IDC->entries() <<G4endl;
		  /*G4cout << "[GateReadout::Digitize]: input hit  \n"
		  		                 <<  *inputDigi << G4endl;
		  G4cout << "[GateReadout::Digitize]: first entrance "
		 		  		                 <<  m_IsFirstEntrance << G4endl;
		    //G4cout<<"Policy "<< m_policy<< " "<< m_depth<<" "<< m_resultingXY<<" "<< m_resultingZ<<Gateendl;
*/
		  const GateOutputVolumeID& blockID = inputDigi->GetOutputVolumeID().Top(m_depth);

		  if (blockID.IsInvalid())
		  {
			  if (nVerboseLevel>1)
			  G4cout << "[GateReadout::Digitize]: out-of-block hit for \n"
					  <<  *inputDigi << Gateendl
					  << " -> digi ignored\n\n";
			  continue;
		   }

		  // Loop inside the temporary output list to see if we have one digi with same blockID as input
		  int this_output_digi = 0;
		  for (this_output_digi=0; this_output_digi<final_nb_out_digi; this_output_digi++)
			  if (final_digi[this_output_digi]->GetOutputVolumeID().Top(m_depth) == blockID) break;

		  // Case: we found an output digi with same blockID
		  if ( this_output_digi!=final_nb_out_digi )
		  {
			  // --------------------------------------------------------------------------------
			  // WinnerTakeAllPolicy (APD like)
			  // --------------------------------------------------------------------------------
			  if (m_policy=="TakeEnergyWinner")
			  {
				  // If energy is higher then replace the digi by the new one
				  if ( inputDigi->GetEnergy() > final_digi[this_output_digi]->GetEnergy() ) final_digi[this_output_digi] = inputDigi;
		          	  final_energy[this_output_digi] += inputDigi->GetEnergy();
		       }
			  // --------------------------------------------------------------------------------
			  // EnergyCentroidPolicy1 (like block PMT)
			  // --------------------------------------------------------------------------------
			  else if (m_policy=="TakeEnergyCentroid") // Crystal element is considered to be the deepest element
			  {
				  // First, if the energy of this digi is higher than the previous one, take it as the reference
		          // in order to have an EnergyWinner policy at levels below the crystal, if any.
		          // The final energy and crystal position will be modified at the end anyway.
		          if ( inputDigi->GetEnergy() > final_digi[this_output_digi]->GetEnergy() ) final_digi[this_output_digi] = inputDigi;
		          // Add the energy to get the total
		          G4double energy = inputDigi->GetEnergy();
		          final_energy[this_output_digi] += energy;
		          // Add the time in order to compute the mean time at the end
		          final_time[this_output_digi] += inputDigi->GetTime();
		          // Get the crystal ID
		          int crystal_id = inputDigi->GetComponentID(m_crystalDepth);
		          // Decompose the crystal_id into X, Y and Z
		          int tmp_crysXY = crystal_id % m_nbCrystalsXY;
		          final_crystal_posZ[this_output_digi] += energy * (((G4double)( crystal_id / m_nbCrystalsXY ))+0.5);
		          final_crystal_posY[this_output_digi] += energy * (((G4double)( tmp_crysXY / m_nbCrystalsX  ))+0.5);
		          final_crystal_posX[this_output_digi] += energy * (((G4double)( tmp_crysXY % m_nbCrystalsX  ))+0.5);
		          // Increment the number of digi contributing to this output digi
		          final_nb_digi[this_output_digi]++;
		        }
		        else
		        {
		          G4Exception( "GateReadout::Digitize", "Digitize", FatalException, "Unknown Readout policy, this is an internal error. Abort.\n");
		        }
		      }
		      // Case: there is no output digi with same blockID
		      else
		      {
		        G4double energy = inputDigi->GetEnergy();
		        if (m_policy=="TakeEnergyCentroid")
		        {
		          // Time will be averaged then
		          final_time[final_nb_out_digi] = inputDigi->GetTime();
		          // Currently there is one digi contributing to this new digi
		          final_nb_digi[final_nb_out_digi] = 1;
		          // Get the crystal ID
		          int crystal_id = inputDigi->GetComponentID(m_crystalDepth);
		          // Decompose the crystal_id into X, Y and Z
		          int tmp_crysXY = crystal_id % m_nbCrystalsXY;
		          final_crystal_posZ[final_nb_out_digi] = energy * (((G4double)( crystal_id / m_nbCrystalsXY ))+0.5);
		          final_crystal_posY[final_nb_out_digi] = energy * (((G4double)( tmp_crysXY / m_nbCrystalsX  ))+0.5);
		          final_crystal_posX[final_nb_out_digi] = energy * (((G4double)( tmp_crysXY % m_nbCrystalsX  ))+0.5);
		        }
		        // Set the current energy
		        final_energy[final_nb_out_digi] += energy;
		        // Store this digi in the list
		        final_digi[final_nb_out_digi] = inputDigi;
		        // Increment the total number of output digi
		        final_nb_out_digi++;
		      }
	  } // End for input digi

	  // S. Stute: create now the output digi list
	  for (int p=0; p<final_nb_out_digi; p++)
	  {
		  // Create the digi
		  m_outputDigi = new GateDigi( final_digi[p] );
		  // Affect energy
		  m_outputDigi->SetEnergy( final_energy[p] );
		  // Special affectations for centroid policy
		  if (m_policy=="TakeEnergyCentroid")
		  {
			  // Affect time being the mean
			  m_outputDigi->SetTime( final_time[p] / ((G4double)final_nb_digi[p]) );
			  // Compute integer crystal indices weighted by total energy
			  G4int crys_posX = ((G4int)(final_crystal_posX[p]/final_energy[p]));
			  G4int crys_posY = ((G4int)(final_crystal_posY[p]/final_energy[p]));
			  G4int crys_posZ = ((G4int)(final_crystal_posZ[p]/final_energy[p]));
			  // Compute final crystal id
			  G4int crystal_id = crys_posZ * m_nbCrystalsXY + crys_posY * m_nbCrystalsX + crys_posX;
			  // We change the level of the volumeID and the outputVolumeID corresponding to the crystal with the new crystal ID
			  m_outputDigi->ChangeVolumeIDAndOutputVolumeIDValue(m_crystalDepth,crystal_id);
			  // Change coordinates (we choose here to place the coordinates at the center of the chosen crystal)
			  //SetGlobalPos(m_system->ComputeObjectCenter(volID));
			  ResetGlobalPos(m_system);
			  ResetLocalPos();
		   }
		  if (nVerboseLevel>1)
			  G4cout << "Created new digi for block " << m_outputDigi->GetOutputVolumeID().Top(m_depth) << ".\n"
				  << "Resulting digi is: \n"
				  << *m_outputDigi << Gateendl << Gateendl ;

		m_OutputDigiCollection->insert(m_outputDigi);
	  }

	  // Free temporary variables used by the centroid policy
	  if (m_policy=="TakeEnergyCentroid")
		  {
			  free(final_time);
		      free(final_crystal_posX);
		      free	(final_crystal_posY);
		      free(final_crystal_posZ);
		      free(final_nb_digi);
		    }
		    // Free other variables
		    free(final_energy);
		    free(final_digi);

		    if (nVerboseLevel==1)
		    {
		    	G4cout << "[GateReadout::Digitize]: returning output digi-list with " << m_OutputDigiCollection->entries() << " entries\n";
		    	for (long unsigned int k=0; k<m_OutputDigiCollection->entries();k++)
		    		{
		    		G4cout << *(*IDC)[k] << Gateendl;
		    		}
		    		G4cout << Gateendl;
		    }

  StoreDigiCollection(m_OutputDigiCollection);

}


// Reset the global position of the pulse with respect to its volumeID that has been changed previously
void GateReadout::ResetGlobalPos(GateVSystem* system)
{
	m_outputDigi->SetGlobalPos(system->ComputeObjectCenter(&(m_outputDigi->m_volumeID)));
}



void GateReadout::DescribeMyself(size_t indent)
{
	G4cout << GateTools::Indent(indent) << " at depth:      " << m_depth << Gateendl;
	G4cout << GateTools::Indent(indent) << "  --> policy: ";
	if (m_policy=="TakeEnergyWinner") G4cout << "TakeEnergyWinner\n";
	else if (m_policy=="TakeEnergyCentroid") G4cout << "TakeEnergyCentroid\n";
	else G4cout << "Unknown policy !\n";
}
