/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
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
    G4cerr << Gateendl << "Warning! Could not install handler for CTRL-\\ (SIGQUIT)!\n" << Gateendl;
    return -1;
  }
  if (signal(SIGXCPU,QuitSignalHandler) == SIG_ERR) {
    G4cerr << Gateendl << "Warning! Could not install handler for SIGXCPU!\n" << Gateendl;
    return -1;
  }
  if (signal(SIGUSR1,QuitSignalHandler) == SIG_ERR) {
    G4cerr << Gateendl << "Warning! Could not install handler for SIGUSR1!\n" << Gateendl;
    return -1;
  }
  if (signal(SIGUSR2,PrintSimulationStatus) == SIG_ERR) {
    G4cerr << Gateendl << "Warning! Could not install handler for SIGUSR2!\n" << Gateendl;
    return -1;
  }
#ifdef __APPLE__
  if (signal(SIGINFO,PrintSimulationStatus) == SIG_ERR) {
    G4cerr << Gateendl << "Warning! Could not install handler for SIGINFO!\n" << Gateendl;
    return -1;
  }
#endif
  return 0;
}


void GateSignalHandler::IgnoreSignalHandler(int sig) {
  G4cerr << "ignoring signal ";
  switch (sig) {
  case SIGXCPU:
    G4cerr << "SIGXCPU\n";
    break;
  case SIGUSR1:
    G4cerr << "SIGUSR1\n";
    break;
  default:
    G4cerr << sig << Gateendl;
    break;
  }
}

// Handles the signal SIGQUIT (CTRL-\).
// When a BeamOn/StartDAQ is running, aborts the current run and stops the DAQ, returning GATE in Idle state.
// In the other states, the signal is ignored.
void GateSignalHandler::QuitSignalHandler(int sig)
{
  // Print that we received the signal to G4cerr
  G4cerr << Gateendl << "Received signal: " ;
  switch (sig)
  {
    case SIGQUIT:
      G4cerr << "Quit (CTRL-\\)\n";
      break;
    default:
      G4cerr << sig << Gateendl;
  }

  // Get the current application state
  G4StateManager *stateManager = G4StateManager::GetStateManager();
  G4ApplicationState state = stateManager->GetCurrentState();
  switch (state)
  {

    case G4State_GeomClosed:
    case G4State_EventProc:

      // If a beamOn/StartDAQ is running, launch abort sequence
      G4cerr << "--- Aborting run/acquisition! ---\n" << Gateendl;

      GateRunManager::GetRunManager()->AbortRun(true);

      GateApplicationMgr::GetInstance()->StopDAQ();
      break;

    case G4State_PreInit:
    case G4State_Init:
    case G4State_Idle:
    case G4State_Quit:
    case G4State_Abort:

    default:
      // If no beamOn/StartDAQ is running, ignore the signal
      G4cerr << "Signal received in state '" << stateManager->GetStateString(state) << "' "
      	     << " --> ignored!\n" << Gateendl;
      break;
  }
}

void GateSignalHandler::PrintSimulationStatus(int)
{
    GateApplicationMgr::GetInstance()->PrintStatus();
}
