/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateARFDataToRoot.hh"
#include "GateARFDataToRootMessenger.hh"
#include "GateVGeometryVoxelStore.hh"

#include "globals.hh"

#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "GateCrystalHit.hh"
#include "GatePhantomHit.hh"
#include "G4VHitsCollection.hh"

#include "G4Trajectory.hh"

#include "G4VProcess.hh"
#include "GateRecorderBase.hh"
#include "G4ios.hh"
#include <iomanip>
#include "G4UImanager.hh"
#include "G4RunManager.hh"
#include "GatePrimaryGeneratorAction.hh"
#include "GateHitConvertor.hh"

#include "G4ParticleDefinition.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"
#include "G4Gamma.hh"
#include "GateApplicationMgr.hh"

#include "GateDigitizer.hh"
#include "GateSingleDigi.hh"

//#include "GateSourceMgr.hh"
#include "GateOutputMgr.hh"

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "G4DigiManager.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateARFDataToRoot::GateARFDataToRoot(const G4String& name, GateOutputMgr* outputMgr,DigiMode digiMode)
  : GateVOutputModule(name,outputMgr,digiMode)
{

  m_isEnabled = false; // Keep this flag false: all output are disabled by default
                       // Moreover this output will slow down a lot all the simulation !
  nVerboseLevel = 0;

//G4cout << " GateARFDataToRoot::GateARFDataToRoot " <<G4endl;

  m_rootMessenger = new GateARFDataToRootMessenger(this);


m_DRFprojectionmode = 0;

m_ARFDatafilename = " "; // All default output file from all output modules are set to " ".
                         // They are then checked in GateApplicationMgr::StartDAQ, using
                         // the VOutputModule pure virtual method GiveNameOfFile()

m_ARFDatafile = 0;
m_ARFDataTree = 0;

m_Xplane = 0.;

NbofGoingOutPhotons = 0;
NbofStraightPhotons = 0;
NbofGoingInPhotons = 0;
NbofKilledInsideCrystalPhotons = 0;
NbofKilledInsideColliPhotons = 0;NbofKilledInsideCamera=0;
NbOfSourcePhotons = 0;
NbofStoredPhotons = 0;
NbOfHeads = 0;
IN_camera = 0;
OUT_camera = 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo...

GateARFDataToRoot::~GateARFDataToRoot() 
{
  delete m_rootMessenger;
  if (nVerboseLevel > 0) G4cout << "GateARFDataToRoot deleting..." << G4endl;

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo...

const G4String& GateARFDataToRoot::GiveNameOfFile()
{
  return m_ARFDatafilename;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo...

// Method called at the beginning of each acquisition by the application manager: opens the ROOT file and prepare the trees
void GateARFDataToRoot::RecordBeginOfAcquisition()
{

   

    m_ARFDatafile = new TFile( m_ARFDatafilename.c_str() ,"RECREATE","ROOT file for ARF purpose");

    m_ARFDataTree = new TTree("theTree","ARF Data Tree");

    m_ARFDataTree->Branch("Edep", &theData.m_Edep,"Edep/D");
    m_ARFDataTree->Branch("outY", &theData.m_Y,"outY/D");
    m_ARFDataTree->Branch("outX", &theData.m_X,"outX/D");

   m_NbOfPhotonsTree = new TTree("theNumberOfPhoton"," statistics on Simulated Photons");
   m_NbOfPhotonsTree->Branch("NOfOutGoingPhot",&NbofGoingOutPhotons,"NbOfOutGoingPhotons/l");
   m_NbOfPhotonsTree->Branch("NbOfInGoingPhot",&NbofGoingInPhotons,"NbOfInGoingPhotons/l");
   m_NbOfPhotonsTree->Branch("NbOfSourcePhot",&NbOfSourcePhotons,"NbOfSourcePhotons/l");
   m_NbOfPhotonsTree->Branch("NbOfInCameraPhot",&IN_camera,"NbOfInCameraPhotons/l");
   m_NbOfPhotonsTree->Branch("NbOfOutCameraPhot",&OUT_camera,"NbOfOutCameraPhotons/l");
   m_NbOfPhotonsTree->Branch("NbOfStoredPhotons",&NbofStoredPhotons,"NbOfStoredPhotons/l");
   m_NbOfPhotonsTree->Branch("NbOfHeads",&NbOfHeads,"NbOfHeads/I");

}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateARFDataToRoot::RecordEndOfAcquisition()
{

CloseARFDataRootFile();
DisplayARFStatistics();
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateARFDataToRoot::RecordBeginOfRun(const G4Run* )
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateARFDataToRoot::RecordEndOfRun(const G4Run* )
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateARFDataToRoot::RecordBeginOfEvent(const G4Event* )
{

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateARFDataToRoot::RecordEndOfEvent(const G4Event* event)
{
RecordDigitizer(event);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateARFDataToRoot::RecordDigitizer(const G4Event* ) 
{
   if (nVerboseLevel > 2)
    G4cout << "GateARFDataToRoot::RecordDigitizer -- begin " << G4endl;

  // Get Digitizer information

  G4DigiManager * fDM = G4DigiManager::GetDMpointer();

  G4int m_collectionID = fDM->GetDigiCollectionID(m_SingleDigiCollectionName);      
  const GateSingleDigiCollection * SDC = 
    (GateSingleDigiCollection*) (fDM->GetDigiCollection( m_collectionID ));

  if (!SDC)
  {
    if (nVerboseLevel>0) 
    G4cout << "[GateARFDataToRoot::SingleOutputChannel::RecordDigitizer]:"<< " digi collection '" << m_SingleDigiCollectionName << "' not found" << G4endl;
    
  } else
   {
    // Digi loop

   if (nVerboseLevel>0)
    G4cout << "[GateARFDataToRoot::SingleOutputChannel::RecordDigitizer]: Total Digits : " 
				     << SDC->entries() <<" in digi collection '" << m_SingleDigiCollectionName << "' " << G4endl;
      G4int n_digi =  SDC->entries();
      for (G4int iDigi=0;iDigi<n_digi;iDigi++) StoreARFData( (*SDC)[iDigi] );// we store the ARF data 

   }
  if (nVerboseLevel > 2)
  G4cout << "GateARFDataToRoot::RecordDigitizer -- end " << G4endl;

}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateARFDataToRoot::RegisterNewSingleDigiCollection(const G4String& aCollectionName,G4bool)
{

  //G4cout << " GateARFDataToRoot::RegisterNewSingleDigiCollection single digi collection name " << aCollectionName<<G4endl;

m_SingleDigiCollectionName = aCollectionName;

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


void GateARFDataToRoot::IncrementNbOfSourcePhotons() 
{
 NbOfSourcePhotons++;
if (nVerboseLevel > 1 && (NbOfSourcePhotons % 1000000) == 0 ) DisplayARFStatistics();
}
                                               
void GateARFDataToRoot::SetARFDataRootFileName(G4String aName)
{ m_ARFDatafilename = aName+".root";}


void GateARFDataToRoot::CloseARFDataRootFile()
{
 m_ARFDatafile = m_ARFDataTree->GetCurrentFile();

 // G4cout << " GateARFDataToRoot::CloseARFDataRootFile : "<< m_ARFDatafile << "   "<< m_NbOfPhotonsTree->GetCurrentFile()<<G4endl;

 

        if ( m_ARFDatafile->IsOpen() ) 
                                {
                                 m_NbOfPhotonsTree->Fill();
                                 
                                 m_ARFDatafile->Write(); 
                                 m_ARFDatafile->Close();
                                }
}

G4int GateARFDataToRoot::StoreARFData(GateSingleDigi* aDigi )
{
G4ThreeVector position = aDigi->GetGlobalPos();
G4ThreeVector PosAtVertex = aDigi->GetSourcePosition();

NbofStoredPhotons++;


G4cout <<" number of stored photons    " << NbofStoredPhotons<<G4endl;
G4cout <<" number of NbOfSourcePhotons "<< NbOfSourcePhotons <<G4endl;

// compute projection of the energy deposition location onto the projection plane as the intersection of the plane X=m_X
// and the line passing through totalposition with unit vector director Indirection, the incident direction

if (m_DRFprojectionmode == 0 ) // smooth positions by substracting source emission position
 {
  theData.m_X = position.z() - PosAtVertex.z();
  theData.m_Y = position.y() - PosAtVertex.y();
 }
  else if ( m_DRFprojectionmode == 1 ) // line project
   {
    G4ThreeVector Indirection = position - PosAtVertex;
    G4double magnitude = Indirection.mag();
    Indirection /= magnitude;
    G4double t = ( m_Xplane - position.x() ) / Indirection.x();
    theData.m_X = position.z() + t * Indirection.z();
    theData.m_Y = position.y() + t * Indirection.y();
   }
    else if ( m_DRFprojectionmode == 2 ) // orthogonal projection
         {
           theData.m_X = position.z();
           theData.m_Y = position.y();
         }

theData.m_Edep =  aDigi->GetEnergy();

m_ARFDataTree->Fill();

return 1;


}


void GateARFDataToRoot::DisplayARFStatistics()
{
G4cout << " Source Photons Statistics For ARF Data " <<  G4endl;
G4cout << " Camera heads number                             " << NbOfHeads<<G4endl;
G4cout << " Source Photons                                  " << NbOfSourcePhotons<<G4endl;
G4cout << " Detected Photons                                " << NbofStoredPhotons <<G4endl;
G4cout << " Source Photons Going Outside the Camera         " << OUT_camera<<G4endl;
G4cout << " Source Photons Going Inside the Camera          " << IN_camera <<G4endl;
G4cout << " Source Photons Going Outside the Crystal        " << NbofGoingOutPhotons<<G4endl;
G4cout << " Source Photons Going Inside the Crystal         " << NbofGoingInPhotons<<G4endl;
G4cout << " Source Photons Killed Inside the Collimator     " << NbofKilledInsideColliPhotons<<G4endl;
G4cout << " Source Photons Killed Inside the Camera         " << NbofKilledInsideCamera<<G4endl;
G4cout << " Source Photons Killed Inside the Crystal        " << NbofKilledInsideCrystalPhotons<<G4endl;

}


void  GateARFDataToRoot::RecordStep(const G4Step*){}

void  GateARFDataToRoot::RecordVoxels(GateVGeometryVoxelStore*){}

#endif







