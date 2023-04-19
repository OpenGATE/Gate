/*
 *	\file Gate.cc
 *	\author Didier Benoit <benoit@imnc.in2p3.fr>
 *	\date May 2012, QIM IMNC-IN2P3/CNRS, Paris VII-XI Universities, Orsay
 *	\version 2.0
 *	\brief To launch GATE:
 *	- 'Gate' or 'Gate --qt' using the Qt visualization
 *	- 'Gate your_macro.mac' or 'Gate --qt your_macro.mac' using the Qt visualization
 *	- 'Gate -d your_macro.mac' using the DigiGate
 *	- 'Gate -a [activity,10]' using the parameterized macro creating an alias in your macro
 */

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include <getopt.h>
#include <cstdlib>
#include <queue>
#include <locale.h>

#include "G4UImanager.hh"
#include "G4UIterminal.hh"
#include "GateUIterminal.hh"
#include "G4UItcsh.hh"
#include "GateRunManager.hh"
#include "GateMessageManager.hh"
#include "GateSteppingVerbose.hh"
#include "GateRandomEngine.hh"
#include "GateApplicationMgr.hh"
#include "GateSourceMgr.hh"
#include "GateSignalHandler.hh"
#include "GateDetectorConstruction.hh"
#include "GatePhysicsList.hh"
#include "GateConfiguration.h"
#include "GateSignalHandler.hh"
#include "GateOutputMgr.hh"
#include "GatePrimaryGeneratorAction.hh"
#include "GateUserActions.hh"
#include "GateDigitizer.hh"
#include "GateClock.hh"
#include "GateUIcontrolMessenger.hh"
#ifdef G4ANALYSIS_USE_ROOT
#include "TPluginManager.h"
#include "GateHitFileReader.hh"
#endif
#ifdef G4VIS_USE
#include "G4VisExecutive.hh"
#endif
#ifdef G4UI_USE
#include "G4UIExecutive.hh"
#ifdef G4UI_USE_QT
#include "qglobal.h"
#if (QT_VERSION >= QT_VERSION_CHECK(4, 0, 0))
#include <G4UIQt.hh>
#include <qmainwindow.h>
#endif
#endif
#endif



//-----------------------------------------------------------------------------
void printHelpAndQuit( G4String msg )
{
  GateMessage( "Core", 0, msg << G4endl );
  GateMessage( "Core", 0, "Usage: Gate [OPTION]... MACRO_FILE" << G4endl );
  GateMessage( "Core", 0, G4endl);
  GateMessage( "Core", 0, "Mandatory arguments to long options are mandatory for short options too." << G4endl );
  GateMessage( "Core", 0, "  -h, --help             print the help" << G4endl );
  GateMessage( "Core", 0, "  -v, --version          print the version" << G4endl );
  GateMessage( "Core", 0, "  -a, --param            set alias. format is '[alias1,value1] [alias2,value2] ...'" << G4endl );
  GateMessage( "Core", 0, "  --d                    use the DigiMode" << G4endl );
  GateMessage( "Core", 0, "  --qt                   use the Qt visualization mode" << G4endl );
  exit( EXIT_FAILURE );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::queue < G4String > decodeParameters( G4String listOfParameters )
{
  // Command queue storing the '/control/alias ALIAS VALUE' command line
  std::queue < G4String > commandQueue;

  // Find the first '[' position and ']' position
  size_t foundBracket1; // '['
  size_t foundBracket2; // ']'
  size_t foundComma; // ','

  foundBracket1 = listOfParameters.find_first_of( "[" );
  foundBracket2 = listOfParameters.find_first_of( "]" );
  foundComma = listOfParameters.find_first_of( "," );

  while( foundBracket1 != G4String::npos )
    {
      // Getting alias
      G4String alias = listOfParameters.substr( foundBracket1 + 1, foundComma - foundBracket1 - 1 );
      // Getting value
      G4String value = listOfParameters.substr( foundComma + 1, foundBracket2 - foundComma - 1 );

      // Creating alias command and store it
      G4String newAliasCommand = G4String( "/control/alias " ) + alias + G4String( " " ) + value;
      commandQueue.push( newAliasCommand );

      // Fetching other bounds []
      foundBracket1 = listOfParameters.find_first_of( "[", foundBracket1 + 1 );
      foundBracket2 = listOfParameters.find_first_of( "]", foundBracket2 + 1 );
      foundComma = listOfParameters.find_first_of( ",", foundComma + 1 );
    }

  return commandQueue;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef G4ANALYSIS_USE_ROOT
void abortIfRootNotFound()
{
  G4cerr  << G4endl
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
          << G4endl;
  G4Exception( "Gate.cc AbortIfRootNotFound", "AbortIfRootNotFound", FatalException, "Correct problem then try again... Sorry!" );
}
#endif
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void executeCommandQueue( std::queue< G4String > commandQueue, G4UImanager* UImanager )
{
  while( commandQueue.size() )
    {
      G4cout << commandQueue.front() << G4endl;
      UImanager->ApplyCommand( commandQueue.front() );
      commandQueue.pop();
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void welcome()
{
  GateMessage("Core", 0, G4endl);
  GateMessage("Core", 0, "*******************************************************" << G4endl);
  GateMessage("Core", 0, " GATE version 9.3 (2023)" << G4endl);
  GateMessage("Core", 0, " Copyright : OpenGATE Collaboration" << G4endl);
  GateMessage("Core", 0, " Reference : Phys. Med. Biol. 49(19) 4543-4561     2004 " << G4endl);
  GateMessage("Core", 0, " Reference : Phys. Med. Biol. 56(4)  881-901       2011 " << G4endl);
  GateMessage("Core", 0, " Reference : Med. Phys.       41(6)  1-14          2014" << G4endl);
  GateMessage("Core", 0, " Reference : Phys. Med. Biol. 66(10) 1-23          2021" << G4endl);
  GateMessage("Core", 0, " http://www.opengatecollaboration.org " << G4endl);
  GateMessage("Core", 0, "*******************************************************" << G4endl);
#ifdef GATE_USE_GPU
  GateMessage("Core", 0, "GPU support activated" << G4endl );
#endif
  GateMessage("Core", 0, G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int main( int argc, char* argv[] )
{

  // First of all, set the G4cout to our message manager
  GateMessageManager* theGateMessageManager = GateMessageManager::GetInstance();
  G4UImanager::GetUIpointer()->SetCoutDestination( theGateMessageManager );

#ifdef G4ANALYSIS_USE_ROOT
  // "Magic" line to avoid problem with ROOT plugin. It is useful when
  // compiling Gate on a given system and executing it remotely on
  // another (grid or cluster).  See
  // http://root.cern.ch/root/roottalk/roottalk08/0690.html
  // DS.
  gROOT->GetPluginManager()->AddHandler( "TVirtualStreamerInfo", "*", "TStreamerInfo", "RIO", "TStreamerInfo()" );
#endif

  GateSteppingVerbose* verbosity = new GateSteppingVerbose;
  G4VSteppingVerbose::SetInstance( verbosity );

  // random engine
  GateRandomEngine* randomEngine = GateRandomEngine::GetInstance();

  // analyzing arguments
  static G4int isDigiMode = 0; // DigiMode false by default
  static G4int isQt = 0; // Enable Qt or not
  G4String listOfParameters = ""; // List of parameters for parameterized macro
  DigiMode aDigiMode = kruntimeMode;


  // Loop over arguments
  G4int c = 0;
  while( 1 )
    {
      // Declaring options
      G4int optionIndex = 0;
      static struct option longOptions[] = {
        { "help", no_argument, 0, 'h' },
        { "version", no_argument, 0, 'v' },
        { "d", no_argument, &isDigiMode, 1 },
        { "qt", no_argument, &isQt, 1 },
        { "param", required_argument, 0, 'a' }
      };

#ifdef __APPLE__
      /*
       * If the program was started by double-clicking on the application bundle on Mac OS X
       * rather than from the command-line, enable Qt and don't try to process other options;
       * argv[1] contains a process serial number in the form -psn_0_1234567
       * OS X <= 10.8 have a -psn_XXX argument given by the system
       * OS X >= 10.9 does not have one, so we use the "TERM" environment variable
       * to distinguish between launched by the Terminal or by the system.
       */
      if ( (argc>1 && strncmp( argv[1], "-psn", 4 ) == 0) || getenv("TERM") == NULL ) {
        argc = 1;
        isQt = 1;
        break;
      }
      else
#endif
        {
          // Getting the option
          c = getopt_long( argc, argv, "hva:", longOptions, &optionIndex );
        }

      // Exit the loop if -1
      if( c == -1 ) break;

      // Analyzing each option
      std::ostringstream ss;
      switch( c )
        {
        case 0:
          // If this option set a flag, do nothing else now
          if( longOptions[ optionIndex ].flag != 0 ) break;
          break;
        case 'h':
          printHelpAndQuit("Gate command line help" );
          break;
        case 'v':
          ss << G4VERSION_MAJOR << "." << G4VERSION_MINOR << "." << G4VERSION_PATCH;
          std::cout << "Gate version is 9.2 ; Geant4 version is " << ss.str() << std::endl;
          exit(0);
          break;
        case 'a':
          listOfParameters = optarg;
          break;
        default:
          printHelpAndQuit( "Out of switch options" );
          break;
        }
    }

  // Checking if the DigiMode is activated
  if( isDigiMode )
    aDigiMode = kofflineMode;

  // Analyzing parameterized macro
  std::queue< G4String > commandQueue = decodeParameters( listOfParameters );

  // Install the signal handler to handle interrupt calls
  GateSignalHandler::Install();

  // Construct the default run manager
  GateRunManager* runManager = new GateRunManager;

  // Set the DetectorConstruction
  GateDetectorConstruction* gateDC = new GateDetectorConstruction();
  runManager->SetUserInitialization( gateDC );

  // Set the PhysicsList
  runManager->SetUserInitialization( GatePhysicsList::GetInstance() );

  // Set the users actions to handle callback for actors - before the initialisation
  new GateUserActions( runManager);

  // Set the Visualization Manager
#ifdef G4VIS_USE
  theGateMessageManager->EnableG4Messages( false );
  G4VisManager* visManager = new G4VisExecutive;
  visManager->Initialize();
  theGateMessageManager->EnableG4Messages( true );
#endif

  // Initialize G4 kernel
  runManager->InitializeAll();

  // Incorporate the user actions, set the particles generator
  runManager->SetUserAction( new GatePrimaryGeneratorAction() );

  // Create various singleton objets
#ifdef G4ANALYSIS_USE_GENERAL
  GateOutputMgr::SetDigiMode( aDigiMode );
  GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
  
  //OK GND 2022. Moved to GateAction:RunAction constructor
  //GateDigitizer* digitizer = GateDigitizer::GetInstance();
  //GatePulseProcessorChain* singleChain = new GatePulseProcessorChain( digitizer, "Singles" );
  //digitizer->StoreNewPulseProcessorChain( singleChain );
#endif

  if( aDigiMode == kofflineMode )
#ifdef G4ANALYSIS_USE_ROOT
    GateHitFileReader::GetInstance();
#else
  abortIfRootNotFound();
#endif

  GateSourceMgr* sourceMgr = GateSourceMgr::GetInstance();
  GateApplicationMgr* appMgr = GateApplicationMgr::GetInstance();
  GateClock::GetInstance()->SetTime( 0 );
  GateUIcontrolMessenger* controlMessenger = new GateUIcontrolMessenger;

  // Get the pointer to the User Interface manager
  G4UImanager* UImanager = G4UImanager::GetUIpointer();

  // Declaring pointers
#ifdef G4UI_USE
  G4UIExecutive* ui = NULL;
#endif
  G4UIsession* session = NULL;
  if( isQt )
    {
#ifdef G4UI_USE
#ifdef G4UI_USE_QT
#if (QT_VERSION >= QT_VERSION_CHECK(4, 0, 0))
      ui = new G4UIExecutive( argc, argv );
      G4UIQt* qui = static_cast<G4UIQt*> (UImanager->GetG4UIWindow());
      if (qui) {
        qui->GetMainWindow()->setVisible(true);
      }
#endif
#endif
#else
#ifdef G4UI_USE_TCSH
      session = new GateUIterminal( new G4UItcsh );
#else
      session = new GateUIterminal();
#endif
#endif
#ifndef _WIN32
      setlocale(LC_NUMERIC, "POSIX");
#endif
    }
  else
    {
#ifdef G4UI_USE_TCSH
      session = new GateUIterminal( new G4UItcsh );
#else
      session = new GateUIterminal();
#endif
    }

  // Macro file parameters
  G4int isMacroFile = 0;
  G4String macrofilename = "";
  // Checking if macro file is here
  // macrofilename always the last arguments, check if '.mac' is in the string
  G4String lastArgument = argv[ argc - 1 ];
  // Finding a point in 'lastArgument'
  size_t foundPoint = lastArgument.find_last_of( "." );
  // Finding suffix
  G4String suffix = "";
  if( foundPoint != G4String::npos )
    suffix = lastArgument.substr( foundPoint + 1 );
  if( suffix == "mac" )
    {
      isMacroFile = 1;
      macrofilename = lastArgument;
    }

  // Using 'session' if not Qt
  welcome();

  std::ostringstream s;
  s << G4VERSION_MAJOR << "." << G4VERSION_MINOR << "." << G4VERSION_PATCH;
  GateMessage( "Core", 0, "You are using Geant4 version " << s.str() << G4endl );

  // Launching Gate if macro file
  if (isMacroFile) {
    executeCommandQueue( commandQueue, UImanager );
    GateMessage( "Core", 0, "Starting macro " << macrofilename << G4endl);
    G4String command = "/control/execute ";
    UImanager->ApplyCommand( command + macrofilename );
    GateMessage( "Core", 0, "End of macro " << macrofilename << G4endl);
  }

#ifdef G4UI_USE
  if (ui) // Launching interactive mode // Qt
    {
      ui->SessionStart();
      delete ui;
    }
  else
#endif
    {
      if (session && !isMacroFile) { // Terminal
        session->SessionStart();
        delete session;
      }
    }


#ifdef G4ANALYSIS_USE_GENERAL
  if (outputMgr) delete outputMgr;
#endif

#ifdef G4VIS_USE
  delete visManager;
#endif

  delete sourceMgr;
  delete appMgr;
  delete randomEngine;
  delete controlMessenger;
  delete verbosity;

  delete runManager;

  return 0;
}
//-----------------------------------------------------------------------------
