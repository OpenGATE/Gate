/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateRunManager.hh"
#include "GateDetectorConstruction.hh"
#include "GateRunManagerMessenger.hh"
#include "GateHounsfieldToMaterialsBuilder.hh"

#include "G4StateManager.hh"
#include "G4UImanager.hh"
#include "G4TransportationManager.hh"
#include "G4PhysListFactory.hh"
#include "G4VUserPhysicsList.hh"
#include "G4RegionStore.hh"
#include "G4Region.hh"


//----------------------------------------------------------------------------------------
GateRunManager::GateRunManager():G4RunManager()
{ 
  pMessenger = new GateRunManagerMessenger(this);
  mHounsfieldToMaterialsBuilder = new GateHounsfieldToMaterialsBuilder();
  mIsGateInitializationCalled = false;
  mUserPhysicList = 0;
  mUserPhysicListName = "";
  EnableGlobalOutput(true);
}
//----------------------------------------------------------------------------------------

GateRunManager::~GateRunManager()
{ 
  delete GateActorManager::GetInstance();
  delete pMessenger;
  delete mHounsfieldToMaterialsBuilder;
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Initialise geometry, actors and the physics list
// This code was derived from the code of G4RunManager::Initialise()
void GateRunManager::InitializeAll()
{  
  DD("GateRunManager::InitializeAll");

  // Get the current application state
  G4StateManager* stateManager = G4StateManager::GetStateManager();
  G4ApplicationState currentState = stateManager->GetCurrentState();

  if (currentState == G4State_PreInit) { DD("G4State_PreInit"); }
  if (currentState == G4State_Init) { DD("G4State_Init"); }
  if (currentState ==  G4State_Idle) { DD(" G4State_Idle"); }

  /*DD(mUserPhysicListName);
  if (mUserPhysicListName != "") {
    G4PhysListFactory *physListFactory = new G4PhysListFactory(); 
    DD("before GetReferencePhysList");
    mUserPhysicList = physListFactory->GetReferencePhysList(mUserPhysicListName); 
    DD("beofre SetUserPhysicList");
    G4RunManager::SetUserInitialization(mUserPhysicList);
  }
*/

  // Check that we're in PreInit or Idle state
  if (currentState!=G4State_PreInit && currentState!=G4State_Idle)
    {
      G4cerr << "Illegal application state - "
	     << "G4RunManager::Initialize() ignored." << G4endl;
      return;
    }
  
  GateMessage("Core", 0, "Initialization of geometry" << G4endl);
  InitGeometryOnly();
      
 // if(!physicsInitialized) {
    GateMessage("Core", 0, "Initialization of physics" << G4endl);
    // We call the PurgeIfFictitious method to delete the gamma related processes
    // that the user defined if the fictitiousProcess is called.
    GatePhysicsList::GetInstance()->PurgeIfFictitious();


    {
      // Check region
      G4RegionStore * RegionStore = G4RegionStore::GetInstance();
      G4RegionStore::const_iterator pi = RegionStore->begin();
      G4RegionStore::const_iterator pe = RegionStore->end();
      while (pi != pe) {
        G4String regionName = (*pi)->GetName();
        DD(regionName);
        ++pi;
      }
    }


    // ------------  //FIXME
    DD(GateRunManager::GetRunManager()->GetUserPhysicsList());
    DD(mUserPhysicListName);
    if (mUserPhysicListName != "") {

      // GatePhysicsList::GetInstance()->DefineCuts();
      //GatePhysicsList::GetInstance()->SetCuts();
      
      G4PhysListFactory *physListFactory = new G4PhysListFactory(); 
      DD("before GetReferencePhysList");
      
      G4ApplicationState currentState = G4StateManager::GetStateManager()->GetCurrentState();
      bool b = G4StateManager::GetStateManager()->SetNewState(G4State_PreInit);
      DD(b);
      mUserPhysicList = physListFactory->GetReferencePhysList(mUserPhysicListName); 

      DD("SetCutsWithDefault");
      mUserPhysicList->SetVerboseLevel(3);
      mUserPhysicList->SetDefaultCutValue(33*mm);      // ok works


      // do not work : 
      mUserPhysicList->SetCutValue(GatePhysicsList::GetInstance()->mapOfRegionCuts["DefaultRegionForTheWorld"].electronCut, "e-");
      mUserPhysicList->SetCutValue(GatePhysicsList::GetInstance()->mapOfRegionCuts["DefaultRegionForTheWorld"].gammaCut, "gamma");
      mUserPhysicList->SetCutValue(GatePhysicsList::GetInstance()->mapOfRegionCuts["DefaultRegionForTheWorld"].positronCut, "e+");
      mUserPhysicList->SetCutValue(GatePhysicsList::GetInstance()->mapOfRegionCuts["DefaultRegionForTheWorld"].protonCut, "proton");

      G4Region * world = (*(G4RegionStore::GetInstance()))[0];
      G4ProductionCuts* pcuts = world->GetProductionCuts();
      pcuts->SetProductionCut(66*mm, "e-");


      mUserPhysicList->SetCutsWithDefault();
      mUserPhysicList->DumpCutValuesTable(); ///-> do nothing
      
      DD("DefineCuts");
      GatePhysicsList::GetInstance()->DefineCuts(mUserPhysicList);

      //      mUserPhysicList->SetCutValue(33*mm, "gamma","DefaultRegionForTheWorld");
  
      /*
      // cut ?
      GatePhysicsList::ParticleCutType worldCuts =     GatePhysicsList::GetInstance()->mapOfRegionCuts["DefaultRegionForTheWorld"];
      //  mUserPhysicList->SetCuts();
      DD(worldCuts.gammaCut);
      DD(worldCuts.electronCut);
      mUserPhysicList->SetCutValue(worldCuts.gammaCut, "gamma","DefaultRegionForTheWorld");
      mUserPhysicList->SetCutValue(worldCuts.electronCut, "e-","DefaultRegionForTheWorld");
      mUserPhysicList->SetCutValue(worldCuts.positronCut, "e+","DefaultRegionForTheWorld");
      mUserPhysicList->SetCutValue(worldCuts.protonCut, "proton","DefaultRegionForTheWorld");
      */

      DD("beofre SetUserPhysicList");
      G4RunManager::SetUserInitialization(mUserPhysicList);
      //GateRunManager::GetRunManager()->SetUserInitialization(mUserPhysicList);
      G4StateManager::GetStateManager()->SetNewState(currentState);

      mUserPhysicList->DumpCutValuesTable(); // dump ?


    }
    // ------------  //FIXME
    DD("before G4RunManager::InitializePhysics()");
    G4RunManager::InitializePhysics();
    DD(GateRunManager::GetRunManager()->GetUserPhysicsList());



    // Check region
    G4RegionStore * RegionStore = G4RegionStore::GetInstance();
    G4RegionStore::const_iterator pi = RegionStore->begin();
    G4RegionStore::const_iterator pe = RegionStore->end();
    while (pi != pe) {
      G4String regionName = (*pi)->GetName();
      DD(regionName);
      ++pi;
    }
      





 // }

  GateMessage("Core", 0, "Initialization of actors" << G4endl);
  GateActorManager::GetInstance() ->CreateListsOfEnabledActors();

  initializedAtLeastOnce = true;
  
  // Set this flag to true (prevent in RunInitialisation to use
  // /run/initialize instead of /gate/run/initialize
  mIsGateInitializationCalled = true;
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Initialise only the geometry, to allow the building of the GATE geometry
// This code was derived from the code of G4RunManager::Initialise()
void GateRunManager::InitGeometryOnly()
{                   


// Initialise G4Regions
 GateDebugMessageInc("Core", 0, "Initialisation of G4Regions" << G4endl);
 G4RegionStore * RegionStore = G4RegionStore::GetInstance();
// RegionStore->Clean();
  G4RegionStore::const_iterator pi = RegionStore->begin();
  while (pi != RegionStore->end()) {
    //G4bool unique;
    G4String regionName = (*pi)->GetName();


    if(regionName!="DefaultRegionForTheWorld"){
       RegionStore->DeRegister((*pi));
      GateMessage("Cuts", 5, "Region "<<regionName<<" deleted."<< G4endl);
    }
    else  ++pi;
  }
  GateMessageDec("Cuts", 5, "G4Regions Initiliazed!" << G4endl);


  // Initialise the geometry in the main() programm
  if (!geometryInitialized) 
    {
      GateMessage("Core", 1, "Initialization of geometry" << G4endl);
      G4RunManager::InitializeGeometry();
    }  
  else
    {
      // Initialize the geometry by calling /gate/run/initialize
      det = detConstruction->GateDetectorConstruction::GetGateDetectorConstruction();
      det->GateDetectorConstruction::SetGeometryStatusFlag(GateDetectorConstruction::geometry_needs_rebuild);
      det->GateDetectorConstruction::UpdateGeometry();	  
      //	  nParallelWorlds = userDetector->ConstructParallelGeometries();
      //          kernel->SetNumberOfParallelWorld(nParallelWorlds);
      //	  geometryInitialized=true;
    }

}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
void GateRunManager::InitPhysics()
{
  DD("GateRunManager::InitPhysics");
  
  G4RunManager::InitializePhysics();
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Overload of G4RunManager()::RunInitialisation() that resets the geometry navigator
void GateRunManager::RunInitialization()
{
  DD("GateRunManager::RunInitialization");

  // Prevent to use /run/initialize instead of /gate/run/initialize
  if (!mIsGateInitializationCalled) {
    GateError("Please, use /gate/run/initialize and not /run/initialize");
  }

  // GateMessage("Core", 0, "Initialization of the run " << G4endl);
  // Perform a regular initialisation
  G4RunManager::RunInitialization();
  
  // Reset the geometry navigator
  // In G4.5, both "/geometry/navigator/reset" and the new method
  // G4RunManager::ResetNavigator() work only in the Idle state,
  // which is incorrect since the geometry is not closed yet
  // This is why we perform a manual reset of the navigator
  G4ThreeVector center(0,0,0);
  G4TransportationManager::GetTransportationManager()
    ->GetNavigatorForTracking()
    ->LocateGlobalPointAndSetup(center,0,false);  

}
//----------------------------------------------------------------------------------------

