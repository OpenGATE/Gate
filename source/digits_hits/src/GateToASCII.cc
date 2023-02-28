/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateToASCII.hh"

#ifdef G4ANALYSIS_USE_FILE

#include "GateToASCIIMessenger.hh"
#include "GateVGeometryVoxelStore.hh"
#include "GateHit.hh"
#include "GatePhantomHit.hh"
#include "GatePrimaryGeneratorAction.hh"
#include "GateVVolume.hh"
#include "GateDigitizerMgr.hh"
#include "GateDigi.hh"
#include "GateCoincidenceDigi.hh"
#include "GateSourceMgr.hh"
#include "GateOutputMgr.hh"

#include "globals.hh"

#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "G4VHitsCollection.hh"
#include "G4TrajectoryContainer.hh"
#include "G4VProcess.hh"
#include "G4ios.hh"
#include "G4UImanager.hh"
#include "G4ParticleDefinition.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"
#include "G4DigiManager.hh"

#include <iomanip>
#include <iostream>
#include <sstream>


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateToASCII::GateToASCII(const G4String& name, GateOutputMgr* outputMgr, DigiMode digiMode)
  : GateVOutputModule(name,outputMgr,digiMode)
  ,m_outFileRunsFlag(digiMode==kruntimeMode)
  ,m_outFileHitsFlag(digiMode==kruntimeMode)
  ,m_outFileVoxelFlag(true)
  ,m_fileName(" ") // All default output file from all output modules are set to " ".
                   // They are then checked in GateApplicationMgr::StartDAQ, using
                   // the VOutputModule pure virtual method GiveNameOfFile()
{
  /*
    if (digiMode==kofflineMode)
    m_fileName=" ";
  */

  m_isEnabled = false; // Keep this flag false: all output are disabled by default
  nVerboseLevel =0;

  m_asciiMessenger = new GateToASCIIMessenger(this);

  GateCoincidenceDigi::SetCoincidenceASCIIMask(1);
  GateDigi::SetSingleASCIIMask(1);

  m_recordFlag = 0; // Design to embrace obsolete functions (histogram, recordVoxels, ...)
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo...

GateToASCII::~GateToASCII()
{
  // for (size_t i=0; i<m_outputChannelList.size() ; ++i )
  //   delete m_outputChannelList[i];

  delete m_asciiMessenger;

  if (nVerboseLevel > 0) G4cout << "GateToASCII deleting...\n";
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

const G4String& GateToASCII::GiveNameOfFile()
{
  return m_fileName;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCII::RecordBeginOfAcquisition()
{
	//OK GND 2022
	 for (size_t i = 0; i < m_outputChannelList.size(); ++i)
	    {
		 m_outputChannelList[i]->m_collectionID=-1 ;
	    }

  if (nVerboseLevel > 2)
    G4cout << "GateToASCII::RecordBeginOfAcquisition\n";

  if (nVerboseLevel > 0) G4cout << "Opening the ASCII output files...";
  if (m_outFileRunsFlag)
    m_outFileRun.open((m_fileName+"Run.dat").c_str(),std::ios::out);
  if (m_outFileHitsFlag)
	{
	  //OK GND 2022
		GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

		m_nSD=digitizerMgr->m_SDlist.size();
	  for (size_t i=0; i<m_nSD ;i++)
		{
			//GateHitTree *treeHit;
		  std::ofstream outFileHits;

			if (digitizerMgr->m_SDlist.size() ==1 ) // keep the old name "Hits" if there is only one collection
				 outFileHits.open((m_fileName+"Hits.dat").c_str(),std::ios::out);
			else
				 outFileHits.open((m_fileName+"Hits_"+ digitizerMgr->m_SDlist[i]->GetName()+".dat").c_str(),std::ios::out);

			m_outFilesHits.push_back(std::move(outFileHits));
		}
	}

  for (size_t i=0; i<m_outputChannelList.size() ; ++i )
    m_outputChannelList[i]->Open(m_fileName);

  if (nVerboseLevel > 0) G4cout << " ... ASCII output files opened\n";
}




//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


void GateToASCII::RecordEndOfAcquisition()
{
  if (nVerboseLevel > 2)
    G4cout << "GateToASCII::RecordEndOfAcquisition\n";
  // Close the file with the hits information
  if (m_outFileRunsFlag)
    m_outFileRun.close();
  if (m_outFileHitsFlag)
  {  //OK GND 2022
	  for (size_t i=0; i< m_nSD;i++)
	  {
		  m_outFilesHits[i].close();
	  }
  }

  for (size_t i=0; i<m_outputChannelList.size() ; ++i )
    m_outputChannelList[i]->Close();

}



//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCII::RecordBeginOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateToASCII::RecordBeginOfRun\n";
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCII::RecordEndOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateToASCII::RecordEndOfRun\n";
  if (m_outFileRunsFlag) {
    G4int nEvent = ((GatePrimaryGeneratorAction*)GateRunManager::GetRunManager()->
		    GetUserPrimaryGeneratorAction())->GetEventNumber();
    if (nVerboseLevel > 0) G4cout
                             << "GateToASCII::RecordEndOfRun: Events in the past run: " << nEvent << Gateendl;
    m_outFileRun
      << " " << std::setw(9) << nEvent
      << Gateendl;
  }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCII::RecordBeginOfEvent(const G4Event* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateToASCII::RecordBeginOfEvent\n";
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCII::RecordEndOfEvent(const G4Event* event)
{
  if (nVerboseLevel > 2)
    G4cout << "GateToASCII::RecordEndOfEvent\n";

  if (m_outFileHitsFlag) {

	//OK GND 2022
	  std::vector<GateHitsCollection*> CHC_vector = GetOutputMgr()->GetHitCollections();

	  for (size_t i=0; i<CHC_vector.size();i++ )//HC_vector.size()
	  {
		  GateHitsCollection* CHC = CHC_vector[i];
		  G4int NbHits = 0;

			if (CHC) {

		   // Hits loop

		    NbHits = CHC->entries();
		    for (G4int iHit=0;iHit<NbHits;iHit++) {
			G4String processName = (*CHC)[iHit]->GetProcess();
			G4int PDGEncoding  = (*CHC)[iHit]->GetPDGEncoding();
			if (nVerboseLevel > 2) G4cout
										 << "GateToASCII::RecordEndOfEvent : HitsCollection: processName : <" << processName
										 << ">    Particls PDG code : " << PDGEncoding << Gateendl;
			if ((*CHC)[iHit]->GoodForAnalysis()) {
			  if (m_outFileHitsFlag) m_outFilesHits[i]  << (*CHC)[iHit];
			}
			  }

			}
			else{
			  if (nVerboseLevel>0) G4cout << "GateToASCII::RecordHits : GateHitCollection not found\n";
			}
		  }
  }


  RecordDigitizer(event);

}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCII::RecordDigitizer(const G4Event* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateToASCII::RecordDigitizer\n";

  for (size_t i=0; i<m_outputChannelList.size() ; ++i )
  {
	  //OK GND 2022
	  if(m_outputChannelList[i]->m_collectionID<0)
	     m_outputChannelList[i]->m_collectionID=GetCollectionID(m_outputChannelList[i]->m_collectionName);
	  m_outputChannelList[i]->RecordDigitizer();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCII::RecordStepWithVolume(const GateVVolume * /*v //WARNING: parameter not used*/, const G4Step* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateToASCII::RecordStep\n";
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCII::RecordVoxels(GateVGeometryVoxelStore* voxelStore)
{
	// TODO !!! OK GND 2020 add (or remove) to GND and documentation

  if (nVerboseLevel > 2)
    G4cout << "[GateToASCII::RecordVoxels]\n";
  if (m_recordFlag>0)
    {
      // protect against huge ASCII files in case of nx,ny,nz ~O(100)
      if (!m_outFileVoxelFlag) return;

      std::ofstream  voxelFile; 	      	    //!< Output stream for the voxel density map
      // Open the header file
      G4String voxelFileName = "voxels.dat";
      voxelFile.open(voxelFileName.c_str(),std::ios::out | std::ios::trunc);
      if (!(voxelFile.is_open()))
        {
          G4String msg = "Could not open the voxel file '" + voxelFileName;
          G4Exception( "GateToASCII::RecordVoxels", "RecordVoxels", FatalException, msg);
        }

      // Write the header: number of voxels, voxel dimensions
      G4int nx = voxelStore->GetVoxelNx();
      G4int ny = voxelStore->GetVoxelNy();
      G4int nz = voxelStore->GetVoxelNz();
      G4ThreeVector voxelSize = voxelStore->GetVoxelSize();
      G4double dx = voxelSize.x();
      G4double dy = voxelSize.y();
      G4double dz = voxelSize.z();

      voxelFile << " " << nx << " " << ny << " " << nz << Gateendl;
      voxelFile << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << dx/mm;
      voxelFile << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << dy/mm;
      voxelFile << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << dz/mm;
      voxelFile << Gateendl;

      // Write the content of the voxel matrix
      for (G4int iz=0; iz<nz; iz++) {
        for (G4int iy=0; iy<ny; iy++) {
          for (G4int ix=0; ix<nx; ix++) {
            G4double density = voxelStore->GetVoxelMaterial(ix,iy,iz)->GetDensity()/(gram/cm3);
            //	G4cout << "Material: " << voxelStore->GetVoxelMaterial(ix,iy,iz)->GetName() << "  density: " << density << Gateendl;
            voxelFile << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << density;
          }
          // line break for each voxel line
          voxelFile << Gateendl;
        }
      }

      voxelFile.close();
    }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCII::Reset()
{
  if (nVerboseLevel > 2)
    G4cout << "GateToASCII::Reset\n";
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCII::RegisterNewSingleDigiCollection(const G4String& aCollectionName,G4bool outputFlag)
{
  SingleOutputChannel* singleOutputChannel =
    new SingleOutputChannel(aCollectionName,outputFlag);
  m_outputChannelList.push_back(singleOutputChannel);

  //  G4cout << " GateToASCII::RegisterNewSingleDigiCollection \n";
  m_asciiMessenger->CreateNewOutputChannelCommand(singleOutputChannel);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCII::RegisterNewCoincidenceDigiCollection(const G4String& aCollectionName,G4bool outputFlag)
{
  CoincidenceOutputChannel* coincOutputChannel =
    new CoincidenceOutputChannel(aCollectionName,outputFlag);
  m_outputChannelList.push_back(coincOutputChannel);

  //  G4cout << " GateToASCII::RegisterNewCoincidenceDigiCollection \n";
  m_asciiMessenger->CreateNewOutputChannelCommand(coincOutputChannel);
}

long GateToASCII::VOutputChannel::m_outputFileSizeLimit = 2000000000;

void GateToASCII::VOutputChannel::Open(const G4String& aFileBaseName)
{
  // if it's not the first file with the same name, add a suffix like _01 to the file name, before .dat
  if ((m_fileCounter > 0) && (m_fileBaseName != aFileBaseName)) {
    m_fileCounter = 0;
  }

  G4String fileCounterSuffix;
  if (m_fileCounter > 0) {
    G4String fileCounterString;
    char buffer [10];
    sprintf(buffer,"%d",m_fileCounter);
    fileCounterString = buffer;
    fileCounterSuffix = G4String("_") + fileCounterString;
  } else {
    fileCounterSuffix = G4String("");
  }
  G4String fileName = aFileBaseName + m_collectionName + fileCounterSuffix + ".dat";
  if (m_outputFlag) {
    m_outputFile.open(fileName,std::ios::out);
    //LF
    //m_outputFile.seekp (0, ios::beg);
    m_outputFile.seekp (0, std::ios::beg);
    //LF
    m_outputFileBegin = m_outputFile.tellp();
  }
  m_fileBaseName = aFileBaseName;
  m_fileCounter++;
}



void GateToASCII::VOutputChannel::Close()
{
  if (m_outputFlag)
    m_outputFile.close();
}

G4bool GateToASCII::VOutputChannel::ExceedsSize()
{
  // from http://www.cplusplus.com/doc/tutorial/tut6-1.html
  long outputFileEnd;
  //LF
  //m_outputFile.seekp (0, ios::end);
  m_outputFile.seekp (0, std::ios::end);
  //LF
  outputFileEnd = m_outputFile.tellp();
  long size = outputFileEnd - m_outputFileBegin; // in bytes
  //   G4cout << "[GateToASCII::VOutputChannel::ExceedsSize]"
  // 	 << " collectionID: " << m_collectionID
  // 	 << " file limit: " << m_outputFileSizeLimit
  // 	 << " file size: " << size << Gateendl;
  return (size > m_outputFileSizeLimit);
}


GateToASCII::SingleOutputChannel::SingleOutputChannel(  const G4String& aCollectionName,
							G4bool outputFlag)
  : GateToASCII::VOutputChannel( aCollectionName, outputFlag )
{
}


void GateToASCII::SingleOutputChannel::RecordDigitizer()
{
  G4DigiManager * fDM = G4DigiManager::GetDMpointer();
  /*if (m_collectionID<0)
    m_collectionID = fDM->GetDigiCollectionID(m_collectionName);
  */
  const GateDigiCollection * SDC =
    (GateDigiCollection*) (fDM->GetDigiCollection( m_collectionID ));

  if (!SDC) {
    if (nVerboseLevel>0) G4cout << "[GateToASCII::SingleOutputChannel::RecordDigitizer]: "
                                << "digi collection '" << m_collectionName <<"' not found\n";
  } else {
    // Digi loop
    if (nVerboseLevel>0) G4cout << "[GateToASCII::SingleOutputChannel::RecordDigitizer]: Totals digits: "
                                << SDC->entries() << Gateendl;
    if (m_outputFlag) {
      G4int n_digi =  SDC->entries();
      for (G4int iDigi=0;iDigi<n_digi;iDigi++) {
	if (m_outputFileSizeLimit > 10000) { // to protect against the creation of too many files by mistake
	  if (ExceedsSize()) {
	    Close();
	    Open(m_fileBaseName);
	  }
	}
        m_outputFile << (*SDC)[iDigi];
      }
    }

  }

}

GateToASCII::CoincidenceOutputChannel::CoincidenceOutputChannel(const G4String& aCollectionName,
								G4bool outputFlag)
  : GateToASCII::VOutputChannel( aCollectionName, outputFlag)
{
}


void GateToASCII::CoincidenceOutputChannel::RecordDigitizer()
{
  G4DigiManager * fDM = G4DigiManager::GetDMpointer();
  //if (m_collectionID<0)
  //  m_collectionID = fDM->GetDigiCollectionID(m_collectionName);
  GateCoincidenceDigiCollection * CDC =
    (GateCoincidenceDigiCollection*) (fDM->GetDigiCollection( m_collectionID ));

  if (!CDC) {
    if (nVerboseLevel>0) G4cout << "[GateToASCII::CoincidenceOutputChannel::RecordDigitizer]: "
                                << "digi collection '" << m_collectionName <<"' not found\n";
  } else {
    // Digi loop
    if (nVerboseLevel>0) G4cout << "[GateToASCII::CoincidenceOutputChannel::RecordDigitizer]: Totals digits: "
                                << CDC->entries() << Gateendl;

    if (m_outputFlag) {
      G4int n_digi =  CDC->entries();
      for (G4int iDigi=0;iDigi<n_digi;iDigi++) {
	if (m_outputFileSizeLimit > 10000) { // to protect against the creation of too many files by mistake
	  if (ExceedsSize()) {
	    Close();
	    Open(m_fileBaseName);
	  }
	}
	m_outputFile << (*CDC)[iDigi];
      }
    }
  }


}

#endif
