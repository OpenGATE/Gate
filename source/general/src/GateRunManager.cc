/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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
#include "G4VModularPhysicsList.hh"
#include "G4RegionStore.hh"
#include "G4LossTableManager.hh"
#include "G4EmStandardPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4EmStandardPhysics_option1.hh"
#include "G4EmStandardPhysics_option2.hh"
#include "G4EmStandardPhysics_option3.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmStandardPhysicsSS.hh"
#include "G4EmLowEPPhysics.hh"
#include "G4EmLivermorePolarizedPhysics.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4EmPenelopePhysics.hh"
#include "G4EmDNAPhysics.hh"
#include "G4StepLimiterPhysics.hh"

//----------------------------------------------------------------------------------------
GateRunManager::GateRunManager() : G4RunManager() {
    pMessenger = new GateRunManagerMessenger(this);
    mHounsfieldToMaterialsBuilder = new GateHounsfieldToMaterialsBuilder();
    mIsGateInitializationCalled = false;
    mUserPhysicList = 0;
    mUserPhysicListName = "";
    mEnableDecay = false;
    EnableGlobalOutput(true);
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateRunManager::~GateRunManager() {
    delete GateActorManager::GetInstance();
    delete pMessenger;
    delete mHounsfieldToMaterialsBuilder;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Initialise geometry, actors and the physics list
// This code was derived from the code of G4RunManager::Initialise()
void GateRunManager::InitializeAll() {
    // Get the current application state
    G4StateManager *stateManager = G4StateManager::GetStateManager();
    G4ApplicationState currentState = stateManager->GetCurrentState();

    // Check that we're in PreInit or Idle state
    if (currentState != G4State_PreInit && currentState != G4State_Idle) {
        GateError("Illegal application state - "
                          << "GateRunManager::Initialize() failed.\n");
        /*G4cerr << "Illegal application state - "
           << "G4RunManager::Initialize() ignored.\n";*/
        return;
    }

    GateMessage("Core", 0, "Initialization of geometry\n");
    InitGeometryOnly();

    // if(!physicsInitialized) {
    GateMessage("Core", 0, "Initialization of physics\n");
    // We call the PurgeIfFictitious method to delete the gamma related processes
    // that the user defined if the fictitiousProcess is called.
    GatePhysicsList::GetInstance()->PurgeIfFictitious();

    // Get the build-in physic list if the user ask for it
    // Note the EM-only physic lists has already been build in GatePhysicsList

    // (not very clear why we need to do nothing when name is equal to "")
    if (mUserPhysicListName != "") {

        G4ApplicationState currentState = G4StateManager::GetStateManager()->GetCurrentState();
        G4StateManager::GetStateManager()->SetNewState(G4State_PreInit);

        mUserPhysicList = NULL;
        if (mUserPhysicListName == "emstandard") {
            mUserPhysicList = new G4VModularPhysicsList();
            mUserPhysicList->RegisterPhysics(new G4EmStandardPhysics());
        }
        if (mUserPhysicListName == "emstandard_opt1") {
            mUserPhysicList = new G4VModularPhysicsList();
            mUserPhysicList->RegisterPhysics(new G4EmStandardPhysics_option1());
        }
        if (mUserPhysicListName == "emstandard_opt2") {
            mUserPhysicList = new G4VModularPhysicsList();
            mUserPhysicList->RegisterPhysics(new G4EmStandardPhysics_option2());
        }
        if (mUserPhysicListName == "emstandard_opt3") {
            mUserPhysicList = new G4VModularPhysicsList();
            mUserPhysicList->RegisterPhysics(new G4EmStandardPhysics_option3());
        }
        if (mUserPhysicListName == "emstandard_opt4") {
            mUserPhysicList = new G4VModularPhysicsList();
            mUserPhysicList->RegisterPhysics(new G4EmStandardPhysics_option4());
        }
        if (mUserPhysicListName == "emstandard_SS") {
            mUserPhysicList = new G4VModularPhysicsList();
            mUserPhysicList->RegisterPhysics(new G4EmStandardPhysicsSS());
        }
        if (mUserPhysicListName == "emLowEP") {
            mUserPhysicList = new G4VModularPhysicsList();
            mUserPhysicList->RegisterPhysics(new G4EmLowEPPhysics());
        }
        if (mUserPhysicListName == "emlivermore") {
            mUserPhysicList = new G4VModularPhysicsList();
            mUserPhysicList->RegisterPhysics(new G4EmLivermorePhysics());
        }
        if (mUserPhysicListName == "emlivermore_polar") {
            mUserPhysicList = new G4VModularPhysicsList();
            mUserPhysicList->RegisterPhysics(new G4EmLivermorePolarizedPhysics());
        }
        if (mUserPhysicListName == "empenelope") {
            mUserPhysicList = new G4VModularPhysicsList();
            mUserPhysicList->RegisterPhysics(new G4EmPenelopePhysics());
        }
        if (mUserPhysicListName == "emDNAphysics") {
            mUserPhysicList = new G4VModularPhysicsList();
            mUserPhysicList->RegisterPhysics(new G4EmDNAPhysics());
        }

        if (mUserPhysicList == NULL) {
            // Need to be in PreInit state (cheat)
            G4PhysListFactory *physListFactory = new G4PhysListFactory();
            // Get G4 physics list from the name
            mUserPhysicList = physListFactory->GetReferencePhysList(mUserPhysicListName);
        }

        // decay ?
        if (mEnableDecay)
            mUserPhysicList->RegisterPhysics(new G4RadioactiveDecayPhysics());

        // Check if it exists
        if (mUserPhysicList == NULL) {
            GateError("Error the physic list named '"
                              << mUserPhysicListName
                              << "' is unknown. The set of available physic list is : "
                              << " emstandard"
                              << " emstandard_opt1"
                              << " emstandard_opt2"
                              << " emstandard_opt3"
                              << " emstandard_opt4"
                              << " emstandard_SS"
                              << " emstandard_GS"
                              << " emLowEP"
                              << " emlivermore"
                              << " emlivermore_polar"
                              << " empenelope"
                              << " emDNAphysics"
                              << GatePhysicsList::GetInstance()->GetListOfPhysicsLists());
        }

        // There are two phys list :
        // - GatePhysicsList::GetInstance() is the one of Gate
        // - mUserPhysicList : the one created by G4 if user choose a buildin PL.
        // the Gate one is used to store/retrieve cuts parameters.

        // Consider the e- cut as default (in mm)
        double def = GatePhysicsList::GetInstance()->mapOfRegionCuts["DefaultRegionForTheWorld"].electronCut;
        mUserPhysicList->SetDefaultCutValue(def * mm);
        mUserPhysicList->SetCutsWithDefault();

        // This function set the cuts define by the user to the phys list build by G4
        GatePhysicsList::GetInstance()->DefineCuts(mUserPhysicList);

        // Also set the lower edge energy
        G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(
                GatePhysicsList::GetInstance()->GetLowEdgeEnergy(),
                G4ProductionCutsTable::GetProductionCutsTable()->GetHighEdgeEnergy());

        // Initialization
        GateRunManager::SetUserInitialization(mUserPhysicList);//use inheritance

        //To take into account the user cuts (steplimiter and special cuts)
        mUserPhysicList->RegisterPhysics(new G4StepLimiterPhysics());

        // Re set the initial state
        G4StateManager::GetStateManager()->SetNewState(currentState);

    } // End if (mUserPhysicListName != "")

    // InitializePhysics
    GateRunManager::InitializePhysics();//use inheritance

    // Take into account the em option set by the user (dedx bin etc)
    GatePhysicsList::GetInstance()->SetEmProcessOptions();

    // Actors initialization
    GateMessage("Core", 0, "Initialization of actors\n");
    GateActorManager::GetInstance()->CreateListsOfEnabledActors();

    initializedAtLeastOnce = true;

    // Set this flag to true (prevent in RunInitialisation to use
    // /run/initialize instead of /gate/run/initialize
    mIsGateInitializationCalled = true;
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Initialise only the geometry, to allow the building of the GATE geometry
// This code was derived from the code of G4RunManager::Initialise()
void GateRunManager::InitGeometryOnly() {
    // Initialise G4Regions
    GateDebugMessageInc("Core", 0, "Initialisation of G4Regions\n");
    G4RegionStore *RegionStore = G4RegionStore::GetInstance();
    // RegionStore->Clean();
    G4RegionStore::const_iterator pi = RegionStore->begin();
    while (pi != RegionStore->end()) {
        //G4bool unique;
        G4String regionName = (*pi)->GetName();

        if (regionName != "DefaultRegionForTheWorld") {
            RegionStore->DeRegister((*pi));
            GateMessage("Cuts", 5, "Region " << regionName << " deleted." << Gateendl);
        } else ++pi;
    }
    GateMessageDec("Cuts", 5, "G4Regions Initialized!\n");

    // Initialise the geometry in the main() programm
    if (!geometryInitialized) {
        GateMessage("Core", 1, "Initialization of geometry\n");
        GateRunManager::InitializeGeometry();//use inheritance
    } else {
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
void GateRunManager::InitPhysics() {
    GateRunManager::InitializePhysics(); //use inheritance
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Overload of G4RunManager()::RunInitialisation() that resets the geometry navigator
void GateRunManager::RunInitialization() {
    // Prevent to use /run/initialize instead of /gate/run/initialize
    if (!mIsGateInitializationCalled) {
        GateError("Please, use /gate/run/initialize and not /run/initialize");
    }

    // GateMessage("Core", 0, "Initialization of the run \n");
    // Perform a regular initialisation
    G4RunManager::RunInitialization();

    // Initialization of the atom deexcitation processes
    // must be done after all other initialization
    if (G4LossTableManager::Instance()->AtomDeexcitation()) {
        G4LossTableManager::Instance()->AtomDeexcitation()->InitialiseAtomicDeexcitation();
    }

    // Reset the geometry navigator
    // In G4.5, both "/geometry/navigator/reset" and the new method
    // G4RunManager::ResetNavigator() work only in the Idle state,
    // which is incorrect since the geometry is not closed yet
    // This is why we perform a manual reset of the navigator
    G4ThreeVector center(0, 0, 0);
    G4TransportationManager::GetTransportationManager()
            ->GetNavigatorForTracking()
            ->LocateGlobalPointAndSetup(center, 0, false);
}
//----------------------------------------------------------------------------------------
