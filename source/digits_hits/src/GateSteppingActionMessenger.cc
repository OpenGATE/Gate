/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateSteppingActionMessenger.hh"
#include "GateActions.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4ios.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

GateSteppingActionMessenger::GateSteppingActionMessenger(GateSteppingAction * msa)
:myAction(msa)
{
  GateSteppingDir = new G4UIdirectory("/gate/stepping/");
  GateSteppingDir->SetGuidance("GATE stepping action control.");

  drawTrajectoryLevelCmd = new G4UIcmdWithAnInteger("/gate/stepping/setDrawTrajectoryLevel",this);
  drawTrajectoryLevelCmd->SetGuidance("Level of trajectory drawing:");
  drawTrajectoryLevelCmd->SetGuidance("0: never");
  drawTrajectoryLevelCmd->SetGuidance("1: for the first 10 events in the run");
  drawTrajectoryLevelCmd->SetGuidance("2: always");
  drawTrajectoryLevelCmd->SetParameterName("N",true);
  drawTrajectoryLevelCmd->SetDefaultValue(0);
  drawTrajectoryLevelCmd->SetRange("N>=0");

  VerboseCmd = new G4UIcmdWithAnInteger("/gate/stepping/verbose",this);
  VerboseCmd->SetGuidance("Set GATE stepping action verbose level");
  VerboseCmd->SetParameterName("verbose",false);
  VerboseCmd->SetRange("verbose>=0");

  SetModeCmd = new G4UIcmdWithAString("/gate/stepping/SetMode",this);

  PolicyCmd = new G4UIcmdWithAString("/gate/stepping/SetPolicy",this);
  PolicyCmd->SetGuidance("Only In Tracker Mode :");
  PolicyCmd->SetGuidance("Set policies :  - StopOnPhantomBoundary : the particles are tracked until they stop on Phantom Boundary");
  PolicyCmd->SetGuidance("                       - StopAfterPhantomBoundary : the particles are traked until they pass Phantom Boundary (Default) ");
  PolicyCmd->SetGuidance("                       - KillTrackAndSecondaries : Once a particle reaches Phantom Boundaries Tracking is Stopped and The Secondaries of This Particles Are Killed.(Default)");
  PolicyCmd->SetGuidance("                       - StopAndKill : same as before but the secondaries are Kept Alive.");
  PolicyCmd->SetGuidance("                       - KeepOnlyPrimaries : only Source Particles are stored.");
  PolicyCmd->SetGuidance("                       - KeepOnlyPhotons : only Photons are stored.");
  PolicyCmd->SetGuidance("                       - KeepOnlyElectrons : only Electrons are stored.");
  PolicyCmd->SetGuidance("                       - KeepAll : All Particles (Primary and Secondary) are stored ( Default) ");

  setEnergyTcmd = new G4UIcmdWithADoubleAndUnit("/gate/stepping/SetEnergyThreshold",this);

  GetTxtCmd = new G4UIcmdWithAString("/gate/stepping/SetTextOutput",this);
  GetTxtCmd->SetGuidance("On - Off . In detector Mode only");
  GetTxtCmd->SetGuidance(" If On write Tracks infos to the file \"PostStepInfo.txt\".");

  SetFilesCmd = new G4UIcmdWithAnInteger("/gate/stepping/SetNumberOfTrackerDataFiles",this);
}

GateSteppingActionMessenger::~GateSteppingActionMessenger()
{
  delete drawTrajectoryLevelCmd;
  delete VerboseCmd;
  delete SetModeCmd;
  delete  PolicyCmd;
  delete GetTxtCmd;
  delete SetFilesCmd;
  //delete SetPhFilesCmd;
  //delete SetRSFilesCmd;
  delete setEnergyTcmd;
  delete GateSteppingDir;

}

void GateSteppingActionMessenger::SetNewValue(G4UIcommand * command,G4String newValue)
{
  if ( command == setEnergyTcmd )
  {
     myAction->SetEnergyThreshold( setEnergyTcmd->GetNewDoubleValue(newValue) );
     return;
  }
  if ( command == SetFilesCmd )
  {
    myAction->SetFiles( SetFilesCmd->GetNewIntValue(newValue) );
    return;
  }
  if ( command == GetTxtCmd )
  {
      myAction->SetTxtOut(newValue);
      return;
  }
  if( command == PolicyCmd )
  {
    G4cout << " ======== Stepping Policies =========\n";
    if ( newValue == "StopOnPhantomBoundary" ) {
      myAction->StopOnBoundary(1);
      return;
    }
    if ( newValue == "StopAfterPhantomBoundary" ) {
      myAction->StopOnBoundary(0);
      return;
    }
    myAction->StopAndKill(newValue);
    return;
  }
  if( command == SetModeCmd )
  {
    TrackingMode theMode = TrackingMode::kUnknown;
    if ( newValue == "Tracker"  ) { theMode = TrackingMode::kTracker;}
    if ( newValue == "Both"   ) { theMode = TrackingMode::kBoth;     }
    if ( newValue == "Detector" ) { theMode = TrackingMode::kDetector;   }
    if ( theMode  == TrackingMode::kUnknown ) {
         G4cout << " Gate Application Manager WARNING : The Application mode " << newValue <<" is not known. Switching to Normal Mode ...\n";theMode = TrackingMode::kBoth;
    }
    myAction->SetMode(theMode);
    return;
  } else if( command==drawTrajectoryLevelCmd ) {
     myAction->SetDrawTrajectoryLevel(drawTrajectoryLevelCmd->GetNewIntValue(newValue));
  } else if( command == VerboseCmd ) {
     myAction->SetVerboseLevel(VerboseCmd->GetNewIntValue(newValue));
  }
}
