/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateOutputMgr.hh"
#include "GateVOutputModule.hh"
#include "GateOutputMgrMessenger.hh"
#include "GateConfiguration.h"
#include "GateAnalysis.hh"
#ifdef GATE_USE_OPTICAL
#include "GateFastAnalysis.hh"
#endif

#include "G4DigiManager.hh"
#include "G4UImanager.hh"
#include "G4UserSteppingAction.hh"
#include "G4SteppingManager.hh"
#include "G4RunManager.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4VVisManager.hh"

#include "GateMessageManager.hh"
#include "GateToDigi.hh"
#include "GateToASCII.hh"
#include "GateToBinary.hh"
#include "GateDigitizer.hh"
#include "GateCrystalSD.hh"
#include "GatePhantomSD.hh"
#include "GateHitFileReader.hh"
#include "GateRandomEngine.hh"
#include "GateARFDataToRoot.hh"
#include "GateToRoot.hh"

GateOutputMgr* GateOutputMgr::instance = 0;


// By default, the output-managers are set in run-time mode
DigiMode   GateOutputMgr::m_digiMode= kruntimeMode;


/*
  We add in this constructor all the output modules, derivated
  from GateVOutputModule Class. Each activated module will be called
  to get the datas that it needs. We fill here an array of
  GateVOutputModule pointers.
  Each module is a different output.

  - GateFastAnalysis (fast alternative for GateAnalysis, disabled by default)
  - GateAnalysis
  - GateToDigi
  - GateToASCII
  - GateToRoot
  - GateToRootPlotter
  - GateToLMF
  - GateToBinary

  All of these Output modules are implemented in the same way.
  They all have a Messenger Class.

*/
//--------------------------------------------------------------------------------
GateOutputMgr::GateOutputMgr(const G4String name)
  : nVerboseLevel(0),
    m_messenger(0),
    mName(name),
    m_acquisitionStarted(false),
    m_allowNoOutput(false)
{
  GateMessage("Output",4,"GateOutputMgr() -- begin"<<G4endl);


  m_messenger = new GateOutputMgrMessenger(this);

#ifdef GATE_USE_OPTICAL
  // fastanalysis should come before GateAnalysis. It does not matter then that both are enabled (only a little
  // speed loss).
  if (m_digiMode==kruntimeMode) {
    GateFastAnalysis* gateFastAnalysis = new GateFastAnalysis("fastanalysis", this,m_digiMode);
    AddOutputModule((GateVOutputModule*)gateFastAnalysis);
  }
#endif

  if (m_digiMode==kruntimeMode) {
    GateAnalysis* gateAnalysis = new GateAnalysis("analysis", this,m_digiMode);
    AddOutputModule((GateVOutputModule*)gateAnalysis);
  }

  GateToDigi* gateToDigi = new GateToDigi("digi", this,m_digiMode);
  AddOutputModule((GateVOutputModule*)gateToDigi);

#ifdef G4ANALYSIS_USE_FILE
  GateToASCII* gateToASCII = new GateToASCII("ascii", this, m_digiMode);
  AddOutputModule((GateVOutputModule*)gateToASCII);
  // For BINARY output
  GateVOutputModule* gateToBinary = new GateToBinary( "binary", this,
                                                      m_digiMode );
  AddOutputModule( gateToBinary );
#endif

#ifdef G4ANALYSIS_USE_ROOT
  GateToRoot* gateToRoot = new GateToRoot("root", this,m_digiMode);
  AddOutputModule((GateVOutputModule*)gateToRoot);
  GateARFDataToRoot* gateARFDataToRoot = new GateARFDataToRoot("arf", this,m_digiMode);
  AddOutputModule((GateVOutputModule*)gateARFDataToRoot);
#endif

  GateMessage("Output",4,"GateOutputMgr() -- end"<<G4endl);
}
//--------------------------------------------------------------------------------

//----------------------------------------------------------------------------------
GateOutputMgr::~GateOutputMgr()
{
  if (m_acquisitionStarted)
    RecordEndOfAcquisition();

  for (size_t iMod = 0; iMod < m_outputModules.size(); iMod++) {
    delete m_outputModules[iMod];
  }
  m_outputModules.clear();
  delete m_messenger;

  if (nVerboseLevel > 0) G4cout << "GateOutputMgr deleting..." << G4endl;

}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::AddOutputModule(GateVOutputModule* module)
{
  if (nVerboseLevel > 2)
    G4cout << "GateOutputMgr::AddOutputModule" << G4endl;

  m_outputModules.push_back(module);
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::RecordBeginOfEvent(const G4Event* event)
{
  GateMessage("Output", 5, "GateOutputMgr::RecordBeginOfEvent" << G4endl;);


  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++) {
    if ( m_outputModules[iMod]->IsEnabled() )
      m_outputModules[iMod]->RecordBeginOfEvent(event);
  }
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::RecordEndOfEvent(const G4Event* event)
{
  GateMessage("Output", 5, "GateOutputMgr::RecordEndOfEvent" << G4endl;);

#ifdef G4ANALYSIS_USE_ROOT
  if (m_digiMode==kofflineMode)
    GateHitFileReader::GetInstance()->PrepareEndOfEvent();
#endif

  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++) {
    if ( m_outputModules[iMod]->IsEnabled() )
      {
        m_outputModules[iMod]->RecordEndOfEvent(event);
      }
  }
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::RecordBeginOfRun(const G4Run* run)
{
  GateMessage("Output", 5, "GateOutputMgr::RecordBeginOfRun" << G4endl;);

  // If the verbosity for the random engine is set, we call the status method
  GateRandomEngine* theRandomEngine = GateRandomEngine::GetInstance();
  if (theRandomEngine->GetVerbosity()>=2) theRandomEngine->ShowStatus();

  if (nVerboseLevel > 2)
    G4cout << "GateOutputMgr::RecordBeginOfRun" << G4endl;

  if (!m_acquisitionStarted)
    RecordBeginOfAcquisition();

  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++) {
    if ( m_outputModules[iMod]->IsEnabled() )
      m_outputModules[iMod]->RecordBeginOfRun(run);
  }
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::RecordEndOfRun(const G4Run* run)
{
  GateMessage("Output", 5, "GateOutputMgr::RecordEndOfRun" << G4endl;);

  if (nVerboseLevel > 2)
    G4cout << "GateOutputMgr::RecordEndOfRun" << G4endl;

  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++) {
    if ( m_outputModules[iMod]->IsEnabled() )
      m_outputModules[iMod]->RecordEndOfRun(run);
  }
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::RecordBeginOfAcquisition()
{
  GateMessage("Output", 5, " GateOutputMgr::RecordBeginOfAcquisition " << G4endl;);

  if (nVerboseLevel > 2)
    G4cout << "GateOutputMgr::RecordBeginOfAcquisition" << G4endl;

#ifdef G4ANALYSIS_USE_ROOT
  if (m_digiMode==kofflineMode)
    GateHitFileReader::GetInstance()->PrepareAcquisition();
#endif


  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++) {
    if ( m_outputModules[iMod]->IsEnabled() )
      m_outputModules[iMod]->RecordBeginOfAcquisition();
  }
  m_acquisitionStarted = true;

  // Start the timer
  m_timer.Start();

}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::RecordEndOfAcquisition()
{

  GateMessage("Output", 5, "GateOutputMgr::RecordEndOfAcquisition" << G4endl;);

  if (nVerboseLevel > 2)
    G4cout << "GateOutputMgr::RecordEndOfAcquisition" << G4endl;

  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++) {
    if ( m_outputModules[iMod]->IsEnabled() )
      m_outputModules[iMod]->RecordEndOfAcquisition();
  }

#ifdef G4ANALYSIS_USE_ROOT
  if (m_digiMode==kofflineMode)
    GateHitFileReader::GetInstance()->TerminateAfterAcquisition();
#endif

  m_acquisitionStarted = false;

  // Stop the time
  m_timer.Stop();
  if (nVerboseLevel > 1) {
    G4cout << "     User simulation time (sec)   := " << (m_timer.GetUserElapsed()) << G4endl;
    G4cout << "     Real simulation time (sec)   := " << (m_timer.GetRealElapsed()) << G4endl;
    G4cout << "     System simulation time (sec) := " << (m_timer.GetSystemElapsed()) << G4endl;
  }

}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::RecordStepWithVolume(const GateVVolume * v, const G4Step* step)
{
  GateMessage("Output", 5, " GateOutputMgr::RecordStep" << G4endl;);
  if (nVerboseLevel > 2)
    G4cout << "GateOutputMgr::RecordStep" << G4endl;

  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++) {
    if ( m_outputModules[iMod]->IsEnabled() )
      m_outputModules[iMod]->RecordStepWithVolume(v, step);
  }
}
//----------------------------------------------------------------------------------

//----------------------------------------------------------------------------------
void GateOutputMgr::RecordVoxels(GateVGeometryVoxelStore* voxelStore)
{
  GateMessage("Output", 5 , " GateOutputMgr::RecordVoxels " << G4endl;);

  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++) {
    if ( m_outputModules[iMod]->IsEnabled() )
      m_outputModules[iMod]->RecordVoxels(voxelStore);
  }
}
//----------------------------------------------------------------------------------

//----------------------------------------------------------------------------------
void GateOutputMgr::Describe(size_t /*indent*/)
{
  G4cout << "GateOutputMgr name: " << mName << G4endl;
  G4cout << "Number of output modules inserted: " << m_outputModules.size() << G4endl;
  G4cout << "Description of the single modules: " << G4endl;

  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++) {
    m_outputModules[iMod]->Describe();
  }
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
GateCrystalHitsCollection* GateOutputMgr::GetCrystalHitCollection()
{
  GateMessage("Output", 5 , " GateOutputMgr::GetCrystalHitCollection " << G4endl;);

  static G4int crystalCollID=-1;     	  //!< Collection ID for the crystal hits

  G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();
  if (crystalCollID==-1)
    crystalCollID = DigiMan->GetHitsCollectionID(GateCrystalSD::GetCrystalCollectionName());

  GateCrystalHitsCollection* CHC = (GateCrystalHitsCollection*) (DigiMan->GetHitsCollection(crystalCollID));

  return CHC;


}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
GatePhantomHitsCollection* GateOutputMgr::GetPhantomHitCollection()
{
  static G4int m_phantomCollID=-1;     	  //!< Collection ID for the phantom hits

  G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();
  if (m_phantomCollID==-1)
    m_phantomCollID = DigiMan->GetHitsCollectionID(GatePhantomSD::GetPhantomCollectionName());

  GatePhantomHitsCollection* PHC = (GatePhantomHitsCollection*) (DigiMan->GetHitsCollection(m_phantomCollID));

  return PHC;
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
GateSingleDigiCollection* GateOutputMgr::GetSingleDigiCollection(const G4String& collectionName)
{
  G4DigiManager * fDM = G4DigiManager::GetDMpointer();
  G4int  collectionID
    = fDM->GetDigiCollectionID(collectionName);

  return (collectionID>=0) ? (GateSingleDigiCollection*) (fDM->GetDigiCollection( collectionID )) : 0;
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
GateCoincidenceDigiCollection* GateOutputMgr::GetCoincidenceDigiCollection(const G4String& collectionName)
{
  G4DigiManager * fDM = G4DigiManager::GetDMpointer();
  G4int  collectionID
    = fDM->GetDigiCollectionID(collectionName);
  return (collectionID>=0) ? (GateCoincidenceDigiCollection*) (fDM->GetDigiCollection( collectionID ) ) : 0 ;
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::RegisterNewSingleDigiCollection(const G4String& aCollectionName,G4bool outputFlag)
{
  GateMessage("Output", 5, " GateOutputMgr::RegisterNewSingleDigiCollection" << G4endl;);

  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++)
    m_outputModules[iMod]->RegisterNewSingleDigiCollection(aCollectionName,outputFlag);
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::RegisterNewCoincidenceDigiCollection(const G4String& aCollectionName,G4bool outputFlag)
{
  GateMessage("Output", 5, " GateOutputMgr::RegisterNewCoincidenceDigiCollection " << G4endl;);

  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++)
    m_outputModules[iMod]->RegisterNewCoincidenceDigiCollection(aCollectionName,outputFlag);
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
GateVOutputModule* GateOutputMgr::GetModule(G4String aName)
{
  std::vector<GateVOutputModule*>::iterator aIt;
  for ( aIt = m_outputModules.begin(); aIt != m_outputModules.end(); aIt++)
    { if ( (*aIt)->GetName() == aName )
        { return (*aIt);
          break;
        }
    }
  return 0;
}
//----------------------------------------------------------------------------------

//----------------------------------------------------------------------------------
void GateOutputMgr::CheckFileNameForAllOutput()
{
  G4int nbActor = GateActorManager::GetInstance()->GetTheListOfActors().size();
  G4int nbModuleEnabled = 0;
  std::vector<GateVOutputModule*>::iterator aIt;
  for ( aIt = m_outputModules.begin(); aIt != m_outputModules.end(); aIt++)
    {
      if ( (*aIt)->IsEnabled() )
        {
          if ((*aIt)->GiveNameOfFile()==" ") // Filename with a space are the default one
            {
              (*aIt)->Enable(false);
              GateWarning("Output module '"+(*aIt)->GetName()+"' was found enabled but no file name was given !!! Output module is so DISABLED !!");
            }
          else if ((*aIt)->GiveNameOfFile()!="  ") nbModuleEnabled++; // Output modules with nofileName return 2 spaces
        }
    }
  if (nbActor==0 && nbModuleEnabled==0)
    {
      if (m_allowNoOutput) GateWarning("Be careful !! No output module nor actor at all are enabled !!");
      else GateError("No output module nor actor are enabled. This simulation will store no result at all.\n \
                    All output modules and actors have to be explicitly enabled now.\n \
                    However if you want to launch this simulation, use the command /gate/output/allowNoOutput\n");
    }
}
//----------------------------------------------------------------------------------

//----------------------------------------------------------------------------------
void GateOutputMgr::BeginOfRunAction(const G4Run* /*aRun*/)
{
  GateMessage("Output", 5, " GateOutputMgr::BeginOfRunAction" << G4endl;);
  /*
    #ifdef G4ANALYSIS_USE_GENERAL
    // Here we fill the histograms of the Analysis manager
    GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
    outputMgr->RecordBeginOfRun(aRun);
    #endif
  */

  // Open the file for hits and digits of this run
  //  char name[15];
  //  sprintf(name,"Events_%d.dat", aRun->GetRunID());
  //  outFile.open(name,ios::out);
  //  G4cout << "Opening outFile...";
  //  outFile.open("petsim.dat",ios::out);
  //  G4cout << " ... outFile opened" << G4endl;

  // Prepare the visualization
  if (G4VVisManager::GetConcreteInstance()) {
    G4UImanager* UI = G4UImanager::GetUIpointer();
    UI->ApplyCommand("/vis/scene/notifyHandlers");
  }

}
//----------------------------------------------------------------------------------



//----------------------------------------------------------------------------------
void GateOutputMgr::EndOfRunAction(const G4Run* /*aRun*/)
{

  GateMessage("Output", 5, " GateOutputMgr::EndOfRunAction" << G4endl;);
  /*
    #ifdef G4ANALYSIS_USE_GENERAL
    // Here we fill the histograms of the Analysis manager
    GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
    outputMgr->RecordEndOfRun(aRun);
    #endif
  */
  // Run ended, update the visualization
  if (G4VVisManager::GetConcreteInstance()) {
    G4UImanager::GetUIpointer()->ApplyCommand("/vis/viewer/update");
  }

  // Close the file with the hits information
  //  outFile.close();

}

//----------------------------------------------------------------------------------

//----------------------------------------------------------------------------------
void GateOutputMgr::BeginOfEventAction(const G4Event* /*evt*/)
{
  GateMessage("Output", 5, " GateOutputMgr::BeginOfEventAction " << G4endl;);
  /*
    #ifdef G4ANALYSIS_USE_GENERAL
    // Here we fill the histograms of the OutputMgr manager
    GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
    outputMgr->RecordBeginOfEvent(evt);
    #endif
  */
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::EndOfEventAction(const G4Event* /*evt*/)
{
  GateMessage("Output", 5, " GateOutputMgr::EndOfEventAction " << G4endl;);
  /*
    #ifdef G4ANALYSIS_USE_GENERAL
    // Here we fill the histograms of the OutputMgr manager
    // Pre-digitalisation outputMgr (hits)
    GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
    outputMgr->RecordEndOfEvent(evt);
    #endif
  */
}
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
void GateOutputMgr::UserSteppingAction(const GateVVolume * /*v*/, const G4Step * /*theStep*/)
{
  GateMessage("Output", 5, " GateOutputMgr::UserSteppingAction -- begin " << G4endl;);

  /*
    #ifdef G4ANALYSIS_USE_GENERAL
    // Here we fill the histograms of the Analysis manager
    GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
    outputMgr->RecordStepWithVolume(v, theStep);

    GateMessage("Output", 5, " GateOutputMgr  RecordStep " << G4endl;);

    #endif

    // In a few random cases, a particle gets 'stuck' in an
    // an infinite loop in the geometry. It then oscillates until GATE
    // crashes on some out-of-memory error.
    // To prevent this from happening, I've added below a quick fix where
    // particles get killed when their step number gets absurdely high
    if ( theStep->GetTrack()->GetCurrentStepNumber() > 10000 )
    theStep->GetTrack()->SetTrackStatus(fStopAndKill);
  */
  GateMessage("Output", 5, " GateOutputMgr::UserSteppingAction -- end" << G4endl;);
}
//----------------------------------------------------------------------------------



/* PY Descourt 11/12/2008 */

void GateOutputMgr::RecordTracks(GateSteppingAction* mySteppingAction){



  for (size_t iMod=0; iMod<m_outputModules.size(); iMod++) {
    if ( m_outputModules[iMod]->IsEnabled() )
      m_outputModules[iMod]->RecordTracks(mySteppingAction);
  }
  GateMessage("Output", 5, " GateOutputMgr::RecordTracks -- end" << G4endl;);
}
