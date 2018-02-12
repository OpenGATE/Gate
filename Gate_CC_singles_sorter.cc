/*
 *	\file Gate_CC_singles_sorter.cc
 */

#include <G4UImanager.hh>
#include "GateRunManager.hh"
#include "GateUIterminal.hh"
#include "GateSignalHandler.hh"
#include "GateDetectorConstruction.hh"
#include "GateROOTBasicOutput.hh"
#include "GateUserActions.hh"

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  // Usage
  std::ostringstream usage;
  usage << std::endl
        << "Gate_CC_singles_sorter" << std::endl
        << "Gate for Compton Camera" << std::endl
        << "Process singles to provide coincidences" << std::endl
        << "Usage : " << argv[0] << " <singles.root> <coincidences.root> <options.mac>" << std::endl;

  // Get user parameters
  if (argc != 4) {
    std::cout << "Need 4 parameters" << std::endl
              << usage.str() << std::endl;
    exit(0);
  }
  std::string singles_filename = argv[1];
  std::string coinc_filename = argv[2];
  std::string options_filename = argv[3];

  // GATE Initialisation
  GateMessageManager* theGateMessageManager = GateMessageManager::GetInstance();
  G4UImanager::GetUIpointer()->SetCoutDestination( theGateMessageManager );
  GateSignalHandler::Install();
  auto runManager = new GateRunManager;
  auto gateDC = new GateDetectorConstruction();
  runManager->SetUserInitialization(gateDC);
  runManager->SetUserInitialization(GatePhysicsList::GetInstance());
  auto myRecords = new GateROOTBasicOutput;
  new GateUserActions(runManager, myRecords);

  // Read macro file
  G4UImanager* UImanager = G4UImanager::GetUIpointer();
  G4String command = "/control/execute ";
  std::cout << "Reading " << options_filename << " ..." << std::endl;
  UImanager->ApplyCommand(command + options_filename);
  std::cout << "Done" << std::endl;

  // Read singles root file

  // Perform sorter according to options in mac file

  // Write output in coincidences.root

  return 0;
}
//-----------------------------------------------------------------------------
