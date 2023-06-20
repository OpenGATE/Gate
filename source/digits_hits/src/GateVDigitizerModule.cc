
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateVDigitizerModule
	- This class is virtual class to construct DigitizerModules from

	- Use GateDummyDigitizerModule and GateDummyDigitizerModuleMessenger class
	to create your DigitizerModule and its messenger

*/
#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"

#include "GateOutputMgr.hh"

GateVDigitizerModule::GateVDigitizerModule(G4String name, G4String path, GateSinglesDigitizer *digitizer,  GateCrystalSD* SD)
  :G4VDigitizerModule(name),
   GateClockDependent(path),
   m_digitizer(digitizer),
   m_SD(SD)
{

	GateOutputMgr::GetInstance()->RegisterNewSingleDigiCollection(digitizer->GetName()+"_"+ SD->GetName()+"_"+name, false);

}

GateVDigitizerModule::GateVDigitizerModule(G4String name, G4String path)
  :G4VDigitizerModule(name),
   GateClockDependent(path)
 {
 }

GateVDigitizerModule::GateVDigitizerModule(G4String name, G4String path, GateCoincidenceDigitizer *digitizer)
  :G4VDigitizerModule(name),
   GateClockDependent(path),
   m_digitizer(digitizer)
{

	//TODO GateOutputMgr::GetInstance()->RegisterNewCoincidenceDigiCollection(digitizer->GetName()+"_"+name, false);

}







GateVDigitizerModule::~GateVDigitizerModule()
{
}



void GateVDigitizerModule::Describe(size_t indent)
{
	G4cout<<"GateVDigitizerModule::Describe"<<G4endl;
  GateClockDependent::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Attached to:        '" << m_digitizer->GetObjectName() << "'\n";
  G4cout << GateTools::Indent(indent) << "Output:             '" << GetObjectName() << "'\n";
  DescribeMyself(indent);
}

void GateVDigitizerModule::DescribeMyself(size_t indent)
{
;
}


//////////////////
void GateVDigitizerModule::InputCollectionID()
{
	G4cout<<" GateVDigitizerModule::InputCollectionID "<<G4endl;
	GateDigitizerMgr* DigiMan = GateDigitizerMgr::GetInstance();
	G4DigiManager* fDM = G4DigiManager::GetDMpointer();


	G4String type;

	//digitizerMgr/crystal/SinglesDigitizer/Singles
	if (G4StrUtil::contains(m_digitizer->GetObjectName(), "SinglesDigitizer"))
		type="SinglesDigitizer";

	else if (G4StrUtil::contains(m_digitizer->GetObjectName(), "CoincidenceSorter"))
		type="CoincidenceSorter";

	else
		type="CoincidenceDiditizer";

	G4cout<<"digitizer_type "<< type<<G4endl;

	G4String DigitizerName;
	G4String outputCollNameTMP;

	if(type!="SinglesDigitizer")
	{
		DigitizerName=((GateCoincidenceDigitizer*)m_digitizer)->GetName();
		outputCollNameTMP = GetName() +"/"+DigitizerName+"_"+m_SD->GetName();
	}
	else
	{
		DigitizerName=((GateSinglesDigitizer*)m_digitizer)->GetName();
		outputCollNameTMP = GetName() +"/"+DigitizerName;
	}


	//DigiMan->ShowSummary();


	G4int DCID = -1;

	G4cout<<"outputCollNameTMP "<<outputCollNameTMP<<G4endl;
	/*if(DCID<0)
	{
		DCID = fDM->GetDigiCollectionID(outputCollNameTMP);
	}
	G4cout<<"DCID "<<DCID<<G4endl;
*/
	/*G4String InitDMname="DigiInit/"+DigitizerName+"_"+m_SD->GetName();
	//G4cout<<"InitDMname "<<InitDMname<<G4endl;
	G4int InitDMID = fDM->GetDigiCollectionID(InitDMname);
	 */
	/*G4String DigitizerName;
	G4String outputCollNameTMP;
	G4String InitDMname;

	G4cout<<m_digitizer->GetObjectName()<<G4endl;

	/*if (!m_coinDigitizer)//digitizerType!="Singles")
		{
			DigitizerName=m_digitizer->GetName();
			outputCollNameTMP = GetName() +"/"+DigitizerName+"_"+m_SD->GetName();
			InitDMname="DigiInit/"+DigitizerName+"_"+m_SD->GetName();
		}
	else //if (digitizerType=="Coincidences")
		{
			DigitizerName=m_coinDigitizer->GetName();
			outputCollNameTMP = GetName() +"/"+DigitizerName;
			InitDMname="CoinDigiInit/"+DigitizerName;
		}



	//check if this module is the first in this digitizer
	if ( m_digitizer->m_DMlist[0] == this )
	{
		//check if the input collection is from InitDM
		G4cout<<"** "<< m_digitizer->GetInputName()<< " "<< m_digitizer->GetOutputName()<<G4endl;
		if (m_digitizer->GetInputName() == m_digitizer->GetOutputName() )
		{
			DCID=InitDMID;
		}

		else
			{
			G4String inputCollectionName = m_digitizer->GetInputName();
			G4cout<<" inputCollectionName "<<inputCollectionName<<G4endl;
			GateSinglesDigitizer* inputDigitizer;

			if (DigiMan->FindSinglesDigitizer(inputCollectionName))
				inputDigitizer = DigiMan->FindSinglesDigitizer(inputCollectionName);
			else
			{
				inputCollectionName= m_digitizer->GetInputName()+"_"+m_digitizer->m_SD->GetName();
				inputDigitizer = DigiMan->FindSinglesDigitizer(inputCollectionName);

			}
			DCID=inputDigitizer->m_outputDigiCollectionID;
			}

	}
	/*else
	{

		G4cout<<"normally here"<<G4endl;
		G4String inputCollectionName = m_digitizer->GetInputName();//+"_"+m_digitizer->m_SD->GetName();
		G4cout<<"inputCollectionName "<<inputCollectionName<<G4endl;

		GateSinglesDigitizer* inputDigitizer = DigiMan->FindSinglesDigitizer(inputCollectionName);
		G4cout<<"inputDigitizer "<< inputDigitizer->GetName()<<" for "<<inputDigitizer->GetInputName() <<G4endl;

		DCID=inputDigitizer->m_outputDigiCollectionID;

		//sequential
	//	if( )
	//	DCID=DCID-1;
	}
*/


	DCID=1;
	if(DCID<0)
	{
      G4Exception( "GateVDigitizerModule::InputCollectionID", "InputCollectionID", FatalException, "Something wrong with collection ID. Please, contact olga[dot]kochebina[at]cea.fr. Abort.\n");
	}
	//G4cout<<DCID<<G4endl;

	G4cout<<"Input collection ID "<< DCID<<G4endl;

 m_DCID = DCID;

}


GateDigi* GateVDigitizerModule::CentroidMerge(GateDigi* right, GateDigi* output )
{

    // AE : Added in a real pulse no sense
    output->m_Postprocess="NULL";         // PostStep process
    output->m_energyIniTrack=-1;         // Initial energy of the track
    output->m_energyFin=-1;         // final energy of the particle
    output->m_processCreator="NULL";
    output->m_trackID=0;
    //-----------------

    // time: store the minimum time
    output->m_time = std::min ( output->m_time , right->m_time ) ;

    // energy: we compute the sum, but do not store it yet
    // (storing it now would mess up the centroid computations)
    G4double totalEnergy = output->m_energy + right->m_energy;

    if (output->m_sourceEnergy != right->m_sourceEnergy) output->m_sourceEnergy=-1;
    if (output->m_sourcePDG != right->m_sourcePDG) output->m_sourcePDG=0;
    if ( right->m_nCrystalConv > output->m_nCrystalConv ){
        output->m_nCrystalConv 	= right->m_nCrystalConv;
    }
    if ( right->m_nCrystalCompton > output->m_nCrystalCompton ){
        output->m_nCrystalCompton 	= right->m_nCrystalCompton;
    }
    if ( right->m_nCrystalRayleigh > output->m_nCrystalRayleigh ){
        output->m_nCrystalRayleigh 	= right->m_nCrystalRayleigh;
    }

    // Local and global positions: store the controids
    if(totalEnergy>0){
        output->m_localPos  =  ( output->m_localPos  * output->m_energy  + right->m_localPos  * right->m_energy ) / totalEnergy ;
        output->m_globalPos =  ( output->m_globalPos * output->m_energy  + right->m_globalPos * right->m_energy ) / totalEnergy ;
    }
    else{
        output->m_localPos  =  ( output->m_localPos  + right->m_localPos)/2;
        output->m_globalPos =  ( output->m_globalPos + right->m_globalPos)/2 ;
    }

    // Now that the centroids are stored, we can store the energy
    output->m_energy   = totalEnergy;


    // # of compton process: store the max nb
    if ( right->m_nPhantomCompton > output->m_nPhantomCompton )
    {
        output->m_nPhantomCompton 	= right->m_nPhantomCompton;
        output->m_comptonVolumeName = right->m_comptonVolumeName;
    }

    // # of Rayleigh process: store the max nb
    if ( right->m_nPhantomRayleigh > output->m_nPhantomRayleigh )
    {
        output->m_nPhantomRayleigh 	= right->m_nPhantomRayleigh;
        output->m_RayleighVolumeName = right->m_RayleighVolumeName;
    }

    // HDS : # of septal hits: store the max nb
    if ( right->m_nSeptal > output->m_nSeptal )
    {
        output->m_nSeptal 	= right->m_nSeptal;
    }

    // VolumeID: should be identical for both pulses, we do nothing
    // m_scannerPos: identical for both pulses, nothing to do
    // m_scannerRotAngle: identical for both pulses, nothing to do
    // m_outputVolumeID: should be identical for both pulses, we do nothing
    return output;
}



GateDigi* GateVDigitizerModule::MergePositionEnergyWin(GateDigi *right, GateDigi *output)
{


    // AE : Added in a real pulse no sense
    output->m_Postprocess="NULL";         // PostStep process
    output->m_energyIniTrack=0;         // Initial energy of the track
    output->m_energyFin=0;         // final energy of the particle
    output->m_processCreator="NULL";
    output->m_trackID=0;
    //-----------------

    // time: store the minimum time
    output->m_time = std::min ( output->m_time , right->m_time ) ;
    if (output->m_sourceEnergy != right->m_sourceEnergy) output->m_sourceEnergy=-1;
    if (output->m_sourcePDG != right->m_sourcePDG) output->m_sourcePDG=0;
    if ( right->m_nCrystalConv > output->m_nCrystalConv ){
    	output->m_nCrystalConv 	= right->m_nCrystalConv;
    }
    if ( right->m_nCrystalCompton > output->m_nCrystalCompton ){
    	output->m_nCrystalCompton 	= right->m_nCrystalCompton;
    }
    if ( right->m_nCrystalRayleigh > output->m_nCrystalRayleigh ){
    	output->m_nCrystalRayleigh 	= right->m_nCrystalRayleigh;
    }



    if( right->m_energy>output->m_max_energy){
    	output->m_max_energy=right->m_energy;
        // Local and global positions: store the positions
    	output->m_localPos  =   right->m_localPos;
    	output->m_globalPos =   right->m_globalPos;

    }
    //G4cout<<output->m_energy <<" + "<< right->m_energy<<G4endl;
    output->m_energy = output->m_energy + right->m_energy;
    //G4cout<<output->m_energy <<G4endl;

    // # of compton process: store the max nb
    if ( right->m_nPhantomCompton > output->m_nPhantomCompton )
    {
    	output->m_nPhantomCompton 	= right->m_nPhantomCompton;
    	output->m_comptonVolumeName = right->m_comptonVolumeName;
    }

    // # of Rayleigh process: store the max nb
    if ( right->m_nPhantomRayleigh > output->m_nPhantomRayleigh )
    {
    	output->m_nPhantomRayleigh 	= right->m_nPhantomRayleigh;
    	output->m_RayleighVolumeName = right->m_RayleighVolumeName;
    }

    // HDS : # of septal hits: store the max nb
    if ( right->m_nSeptal > output->m_nSeptal )
    {
    	output->m_nSeptal 	= right->m_nSeptal;
    }

    // VolumeID: should be identical for both pulses, we do nothing
    // m_scannerPos: identical for both pulses, nothing to do
    // m_scannerRotAngle: identical for both pulses, nothing to do
    // m_outputVolumeID: should be identical for both pulses, we do nothing

    return output;
}









