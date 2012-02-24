#include "GateRunManager.hh"
#include "G4UImanager.hh"

#include <queue>

#include "G4UIterminal.hh"
#include "G4UItcsh.hh"
#include "GateSteppingVerbose.hh"

#ifdef G4UI_USE_ROOT
#include "G4UIRoot.hh"
#endif

#include "GateDetectorConstruction.hh"
#include "GatePhysicsList.hh"
#include "GatePrimaryGeneratorAction.hh"
#include "GateUserActions.hh"
#include "GateRandomEngine.hh"

#include "GateApplicationMgr.hh"
#include "GateSourceMgr.hh"
#include "GateClock.hh"
#include "GateOutputMgr.hh"
#include "GateUIcontrolMessenger.hh"
#include "GateUIterminal.hh"
#include "GateDigitizer.hh"
#include "GatePulseProcessorChain.hh"
#include "GateSignalHandler.hh"

#include "GateRecorderBase.hh"

#ifdef G4ANALYSIS_USE_ROOT
#include "GateROOTBasicOutput.hh"
#include "TPluginManager.h"
#include "GateHitFileReader.hh"
#endif

#include "GateMessageManager.hh"

#ifdef G4VIS_USE
#include "G4VisExecutive.hh"
#endif

#ifndef G4ANALYSIS_USE_ROOT
//-------------------------------------------------------------------------------------
void AbortIfRootNotFound()
{
  G4cout  << G4endl 
	  << "Sorry, but it seems that GATE was compiled without the ROOT option." << G4endl
	  << "Consequently, you can not run GATE in DigiGate mode, and the execution will abort." << G4endl
	  << G4endl
	  << "There maybe several reasons why GATE was compiled without the ROOT option:" << G4endl
	  << G4endl
	  << "1) ROOT is not installed on your system;" << G4endl
	  << "2) You did not source a GATE configuration script before compiling GATE;" << G4endl
	  << "3) The configuration script you used has been modified to disable the ROOT option;" << G4endl
	  << "4) You used the configuration file 'env_gate.csh' but it did not set the ROOT option." << G4endl
	  << G4endl
	  << "Here is what you can do:" << G4endl
	  << G4endl
	  << "1) I'm sorry but you won't have access to DigiGate, as it needs ROOT to work." << G4endl 
	  << "   We apologize for this inconvenience, but there is nothing we can do about it," << G4endl
	  << "   because DigiGate works by re-reading a ROOT hit-file" << G4endl 
	  << G4endl
	  << "2,3) You must enable the ROOT option with:" << G4endl
	  << "         setenv G4ANALYSIS_USE_ROOT 1 (csh)" << G4endl
	  << "         export G4ANALYSIS_USE_ROOT=1 (sh)"  << G4endl
	  << "   Then rebuild the ROOT-dependent part of GATE (see below)" << G4endl 
	  << G4endl
	  << "4) The script 'env_gate.csh' decides whether or not to set the ROOT option by checking the value of " << G4endl
	  << "   the environment variable 'ROOTSYS'. Maybe ROOT is installed on your system but ROOTSYS is not set." << G4endl
	  << "   If that's the case, you must set the variable ROOTSYS with:" << G4endl
	  << "         setenv ROOTSYS the_home_dir_of_ROOT (csh)" << G4endl
	  << "         export ROOTSYS=the_home_dir_of_ROOT (sh)"  << G4endl
	  << "   Then source 'env_gate.csh' again, then rebuild the ROOT-dependent part of GATE (see below)" << G4endl 
	  << G4endl
	  << "Note: to rebuild the ROOT-dependent part of GATE, use the following command:" << G4endl
	  << "         touch Gate.cc `grep -l G4ANALYSIS_USE_ROOT include/* src/*`" << G4endl
	  << "         make" << G4endl
	  << G4endl;
  G4Exception("Correct problem then try again... Sorry!");	    
}
//-------------------------------------------------------------------------------------
#endif

//-------------------------------------------------------------------------------------
// Variables decoded from the Gate arguments
DigiMode aDigiMode=kruntimeMode; 
/* OBSOLETE
G4bool isBatchMode = false;
char* batchFileName = 0;
*/
std::queue<G4String> commandQueue;

//-------------------------------------------------------------------------------------
// Decode the arguments passed to Gate
int nextArg = 1; // Need to be global to be used in main
bool isMacroFile = false;
void DecodeArguments(int argc,char** argv)
{
  // Loop while there are optional arguments on the command line
  while (nextArg<argc)
    {
      if (!strcmp(argv[nextArg],"-d")) {
	// Select DigiGate mode
	aDigiMode=kofflineMode;
	nextArg++;
      }
      else if (!strcmp(argv[nextArg],"-a")) {
	// Pass an alias to Gate
	if ( argc <= (nextArg+2) )
	  G4Exception("Not enough arguments after the option flag '-a'\n"
		      "The syntax to use this flag, which allows to create an alias, is:\n"
		      "\tGate -a ALIAS_NAME ALIAS_VALUE\n\n"
		      "Aborting!");
	//G4String newCommand = G4String("/control/alias ") + argv[++nextArg] + " \"" + argv[++nextArg] + "\"";
	nextArg++ ; 
	G4String newCommand = G4String("/control/alias ") + argv[nextArg] ;
        nextArg++ ; 
	newCommand = newCommand + " \"" + argv[nextArg] + "\"";
	commandQueue.push(newCommand);	    
        nextArg++;
      }
#ifdef G4ANALYSIS_USE_ROOT
      else if (!strcmp(argv[nextArg],"-b")) {
	// disable X connection if asked...
	gROOT->SetBatch();
	nextArg++;
      }
#endif
      else {
	// The argument was not recognised: exit
	isMacroFile = true;
	break;
      }
    }

/* OBSOLETE
  // If there is an argument left on the command line, select the batch mode
  // and use it as the batch-file name
  isBatchMode = (argc!=nextArg);
  if (isBatchMode)
    batchFileName = argv[nextArg];
*/
}
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
// Execute the list of commands obtained from the command-line arguments
void ExecuteCommandQueue()
{
  G4UImanager* anUImanager = G4UImanager::GetUIpointer();

  while ( commandQueue.size() )
    { 
      G4cout << commandQueue.front() << G4endl;
      anUImanager->ApplyCommand(commandQueue.front());
      commandQueue.pop() ; 
    }
}
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
void Welcome() {
  GateMessage("Core", 0, G4endl);
  GateMessage("Core", 0, "**********************************************************************"<< G4endl);
  GateMessage("Core", 0, " GATE version name: gate_v6.1                                         "<< G4endl);
  GateMessage("Core", 0, "                    Copyright : OpenGATE Collaboration                "<< G4endl);
  GateMessage("Core", 0, "                    Reference : Phys. Med. Biol. 49 (2004) 4543-4561  "<< G4endl);
  GateMessage("Core", 0, "                    Reference : Phys. Med. Biol. 56 (2011) 881-901    "<< G4endl);
  GateMessage("Core", 0, "                    WWW : http://www.opengatecollaboration.org/       "<< G4endl);
  GateMessage("Core", 0, "**********************************************************************"<< G4endl);
  GateMessage("Core", 0, G4endl);
}
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
int main(int argc,char** argv)
{
  // First of all, set the G4cout to our message manager
  GateMessageManager * theGateMessageManager = GateMessageManager::GetInstance();
  G4UImanager::GetUIpointer()->SetCoutDestination(theGateMessageManager);

#ifdef G4ANALYSIS_USE_ROOT
  // "Magic" line to avoid problem with ROOT plugin. It is useful when
  // compiling Gate on a given system and executing it remotely on
  // another (grid or cluster).  See
  // http://root.cern.ch/root/roottalk/roottalk08/0690.html
  // DS.
  gROOT->GetPluginManager()->AddHandler("TVirtualStreamerInfo", "*",
                                        "TStreamerInfo", "RIO", "TStreamerInfo()");
#endif
 
 GateSteppingVerbose* verbosity = new GateSteppingVerbose;
 G4VSteppingVerbose::SetInstance(verbosity);



  // random engine
  GateRandomEngine* randomEngine = GateRandomEngine::GetInstance();

  // Call the argument decoding function
  DecodeArguments(argc,argv);

  // Install the signal handler to handle interrupt calls
  GateSignalHandler::Install();

  // Construct the default run manager
  GateRunManager* runManager = new GateRunManager;  

  // Set the Basic ROOT Output
GateRecorderBase* myRecords = 0;
#ifdef G4ANALYSIS_USE_ROOT
  myRecords= new GateROOTBasicOutput;
#endif
  
  // Set the DetectorConstruction
  GateDetectorConstruction* gateDC = new GateDetectorConstruction();
  runManager->SetUserInitialization(gateDC);
  
  // Set the PhysicsList  
  runManager->SetUserInitialization(GatePhysicsList::GetInstance());

  // Set the users actions to handle callback for actors - before the initialisation
  new GateUserActions(runManager, myRecords);

  // Set the Visualization Manager
#ifdef G4VIS_USE
  theGateMessageManager->EnableG4Messages(false);
  G4VisManager* visManager = new G4VisExecutive;
  visManager->Initialize();
  theGateMessageManager->EnableG4Messages(true);
#endif
  
  // Initialize G4 kernel
  runManager->InitializeAll();

 // Incorporate the user actions
  // Set the particles generator
  runManager->SetUserAction(new GatePrimaryGeneratorAction());  

  // Set the users actions to handle callback for actors - before the initialisation
  //new GateUserActions(runManager, myRecords);
   

  // Create various singleton objets
#ifdef G4ANALYSIS_USE_GENERAL
  GateOutputMgr::SetDigiMode(aDigiMode);
  GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
  GateDigitizer* digitizer = GateDigitizer::GetInstance();
  GatePulseProcessorChain* singleChain = new GatePulseProcessorChain(digitizer,"Singles"); 
  digitizer->StoreNewPulseProcessorChain(singleChain);
#endif

  if (aDigiMode==kofflineMode)
#ifdef G4ANALYSIS_USE_ROOT
    GateHitFileReader::GetInstance();
#else
  AbortIfRootNotFound();
#endif

  GateSourceMgr*      sourceMgr = GateSourceMgr::GetInstance();
  GateApplicationMgr* appMgr    = GateApplicationMgr::GetInstance();
  GateClock::GetInstance()->SetTime(0);
  GateUIcontrolMessenger* controlMessenger = new GateUIcontrolMessenger;

#ifdef G4VMC
  new TG4XMLMessenger(new TG4XMLGeometryGenerator);
#endif


  //******************************************************
  // Get the pointer to the UI manager and set verbosities
  //******************************************************
  
  // Start the execution of fGATE
  // Two modes are possible: batch or interactive
  G4UIsession* session=0;
 
  if (!isMacroFile)  // Define (G)UI terminal for interactive mode
    {   
#ifdef G4UI_USE_TCSH
      session = new G4UIterminal(new G4UItcsh);      
#else
      session = new G4UIterminal();
#endif
    }

  //******************************************************     
  // Get the pointer to the User Interface manager
  //******************************************************
  G4UImanager* UI = G4UImanager::GetUIpointer(); 

  if (session)   // Define UI session for interactive mode
    {
      Welcome();
      // G4UIterminal is a (dumb) terminal      //
      // UI->ApplyCommand("/control/execute vis.mac");    
           
#ifdef G4UI_USE_ROOT
      // G4UIRoot is a ROOT based GUI.
      GateMessage("Core", 0,  << "Creating the ROOT UI session..." << G4endl);
      session = new G4UIRoot(argc,argv,"GATE", "Geant4 Application for Tomographic Emission","root_logo.xpm", "logoGate_medium.xpm");
#endif
      
      session->SessionStart();
      //delete session;
    }
  else           // Batch mode
    { 
#ifdef G4VIS_USE
      visManager->SetVerboseLevel("quiet");
#endif
      Welcome();      
      ExecuteCommandQueue();
			GateMessage("Core", 0, "Starting macro " << argv[nextArg] << G4endl);
      G4String command = "/control/execute ";
      G4String fileName = argv[nextArg];
      UI->ApplyCommand(command+fileName);
      GateMessage("Core", 0, "End of macro " << fileName << G4endl);
    }
  
  // Job termination 
  // Free the store: user actions, physics_list and
  // detector_description are owned and deleted by the run manager, so
  // they should not be deleted in the main() program !



#ifdef G4ANALYSIS_USE_GENERAL
  if (outputMgr) delete outputMgr;
#endif

#ifdef G4VIS_USE
  delete visManager;
#endif

  delete sourceMgr;
  delete appMgr;
  delete runManager;
  delete randomEngine;
  delete myRecords;
  delete controlMessenger;

  return 0;
}
//-------------------------------------------------------------------------------------


