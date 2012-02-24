/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSignalHandler.hh"

#include <signal.h>

#include "GateRunManager.hh"
#include "GateApplicationMgr.hh"
#include "G4StateManager.hh"

// Install the signal handlers
G4int GateSignalHandler::Install()
{
  // Set the SIGQUIT signal handler to QuitSignalHandler
  if (signal(SIGQUIT,QuitSignalHandler) == SIG_ERR) {
    G4cerr << G4endl << "Warning! Could not install handler for CTRL-\\ (SIGQUIT)!" << G4endl << G4endl;
    return -1;
  }
  return 0;
}





// Handles the signal SIGQUIT (CTRL-\). 
// When a BeamOn/StartDAQ is running, aborts the current run and stops the DAQ, returning GATE in Idle state.
// In the other states, the signal is ignored.
void GateSignalHandler::QuitSignalHandler(int sig)
{
  // Print that we received the signal to G4cerr
  G4cerr << G4endl << "Received signal: " ;
  switch (sig)
  {
    case SIGQUIT:
      G4cerr << "Quit (CTRL-\\)" << G4endl;
      break;
    default:
      G4cerr << sig << G4endl;
  }
  
  // Get the current application state
  G4StateManager *stateManager = G4StateManager::GetStateManager();
  G4ApplicationState state = stateManager->GetCurrentState();
  switch (state)
  {
#ifdef G4VERSION4
    case GeomClosed:
    case EventProc:
#else
    case G4State_GeomClosed:
    case G4State_EventProc:
#endif
      // If a beamOn/StartDAQ is running, launch abort sequence
      G4cerr << "--- Aborting run/acquisition! ---" << G4endl << G4endl;
#ifdef G4VERSION4
      G4RunManager::GetRunManager()->AbortRun();
#else
      G4RunManager::GetRunManager()->AbortRun(true);
#endif
      GateApplicationMgr::GetInstance()->StopDAQ();
      break;
#ifdef G4VERSION4
    case PreInit:
    case Init:
    case Idle:
    case Quit:
    case Abort:
#else
    case G4State_PreInit:
    case G4State_Init:
    case G4State_Idle:
    case G4State_Quit:
    case G4State_Abort:
#endif
    default:
      // If no beamOn/StartDAQ is running, ignore the signal
      G4cerr << "Signal received in state '" << stateManager->GetStateString(state) << "' "
      	     << " --> ignored!" << G4endl << G4endl;
      break;
  }
}

