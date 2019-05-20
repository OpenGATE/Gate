/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateApplicationMgrMessenger.hh"
#include "GateApplicationMgr.hh"
#include "GateSourceMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//-------------------------------------------------------------------------------------------------------------------
GateApplicationMgrMessenger::GateApplicationMgrMessenger()
{
  GateApplicationDir = new G4UIdirectory("/gate/application/");
  GateApplicationDir->SetGuidance("Gate application control.");

  TimeSliceCmd = new G4UIcmdWithADoubleAndUnit("/gate/application/setTimeSlice",this);
  TimeSliceCmd->SetGuidance("Set time slice");
  TimeSliceCmd->SetParameterName("Time",false);
  TimeSliceCmd->SetUnitCategory("Time");
  //  TimeSliceCmd->AvailableForStates(Idle);

  TimeStartCmd = new G4UIcmdWithADoubleAndUnit("/gate/application/setTimeStart",this);
  TimeStartCmd->SetGuidance("Set start time for the DAQ");
  TimeStartCmd->SetParameterName("Time",false);
  TimeStartCmd->SetUnitCategory("Time");
  //  TimeStartCmd->AvailableForStates(Idle);

  TimeStopCmd = new G4UIcmdWithADoubleAndUnit("/gate/application/setTimeStop",this);
  TimeStopCmd->SetGuidance("Set stop time for the DAQ");
  TimeStopCmd->SetParameterName("Time",false);
  TimeStopCmd->SetUnitCategory("Time");
  //  TimeStopCmd->AvailableForStates(Idle);

  AddSliceCmd = new G4UIcmdWithADoubleAndUnit("/gate/application/addSlice",this);
  AddSliceCmd->SetGuidance("Add a new time slice");
  AddSliceCmd->SetParameterName("Time",false);
  AddSliceCmd->SetUnitCategory("Time");

  StartDAQCmd = new G4UIcmdWithoutParameter("/gate/application/startDAQ",this);
  StartDAQCmd->SetGuidance("Start the DAQ");
  //  StartDAQCmd->AvailableForStates(Idle);

  StartCmd = new G4UIcmdWithoutParameter("/gate/application/start",this);
  StartCmd->SetGuidance("Start the simulation");
  //  StartDAQCmd->AvailableForStates(Idle);

  StartDAQCompleteCmd = new G4UIcmdWith3VectorAndUnit("/gate/application/startDAQComplete",this);
  StartDAQCompleteCmd->SetGuidance("Set properties of the acquisition and launch it.");
  StartDAQCompleteCmd->SetGuidance("[usage] /gate/application/startDAQComplete timeStart timeStop timeSlice unit");
  StartDAQCompleteCmd->SetGuidance("        1. timeStart : (double) ");
  StartDAQCompleteCmd->SetGuidance("        2. timeStop  : (double) ");
  StartDAQCompleteCmd->SetGuidance("        3. timeSlice : (double) ");
  StartDAQCompleteCmd->SetGuidance("        4. unit");
  StartDAQCompleteCmd->SetUnitCategory("Time");
  StartDAQCompleteCmd->SetDefaultUnit("s");

  StartDAQClusterCmd = new G4UIcmdWith3VectorAndUnit("/gate/application/startDAQCluster",this);
  StartDAQClusterCmd->SetGuidance("Set properties of the acquisition and launch it \n for GATE cluster approach (gjs).");
  StartDAQClusterCmd->SetGuidance("[usage] /gate/application/startDAQCluster virtualStart virtualStop dummy unit");
  StartDAQClusterCmd->SetGuidance("        1. virtual time Start : (double) ");
  StartDAQClusterCmd->SetGuidance("        2. virtual time Stop  : (double) ");
  StartDAQClusterCmd->SetGuidance("        3. dummy : (double) ");
  StartDAQClusterCmd->SetGuidance("        4. unit");
  StartDAQClusterCmd->SetUnitCategory("Time");
  StartDAQClusterCmd->SetDefaultUnit("s");

  StopDAQCmd = new G4UIcmdWithoutParameter("/gate/application/stopDAQ",this);
  StopDAQCmd->SetGuidance("Stop the DAQ");
  //  StopDAQCmd->AvailableForStates(Idle);

  PauseDAQCmd = new G4UIcmdWithoutParameter("/gate/application/pauseDAQ",this);
  PauseDAQCmd->SetGuidance("Pause the DAQ");
  //  StopDAQCmd->AvailableForStates(Idle);

  //  ExitFlagCmd = new G4UIcmdWithABool("/gate/application/setExitFlag",this);
  //  ExitFlagCmd->SetGuidance("Set GATE application manager exit flag");
  //  ExitFlagCmd->SetGuidance("If true, it stops the DAQ loop");

  VerboseCmd = new G4UIcmdWithAnInteger("/gate/application/verbose",this);
  VerboseCmd->SetGuidance("Set GATE application manager verbose level");
  VerboseCmd->SetParameterName("verbose",false);
  VerboseCmd->SetRange("verbose>=0");

  DescribeCmd = new G4UIcmdWithoutParameter("/gate/application/describe",this);
  DescribeCmd->SetGuidance("List the DAQ parameters");
  //  DescribeCmd->AvailableForStates(Idle);

  NoOutputCmd = new G4UIcmdWithoutParameter("/gate/application/noGlobalOutput", this);
  NoOutputCmd->SetGuidance("Supress the global output manager (PET/SPECT), use this macro if you do not need PET/SPECT systems.");

  //EnableSuccessiveSourceMode = new G4UIcmdWithoutParameter("/gate/application/enableSuccessiveSourceMode", this);
  //EnableSuccessiveSourceMode->SetGuidance("Enable 'successive source' mode. Each source will be selected in a successive way (alternative default mode is 'competitive mode').");

  ReadTimeSlicesInAFileCmd = new G4UIcmdWithAString("/gate/application/readTimeSlicesIn", this);
  ReadTimeSlicesInAFileCmd->SetGuidance("Read the different time slices ('run') in a file.");

  SetTotalNumberOfPrimariesCmd = new G4UIcmdWithADouble("/gate/application/setTotalNumberOfPrimaries", this);
  SetTotalNumberOfPrimariesCmd->SetGuidance("Set the total number of primaries to generate in the whole simulation.");

  SetNumberOfPrimariesPerRunCmd = new G4UIcmdWithADouble("/gate/application/setNumberOfPrimariesPerRun", this);
  SetNumberOfPrimariesPerRunCmd->SetGuidance("Set the number of primaries to generate per per run.");

  SetNumberOfPrimariesPerRunCmd2 = new G4UIcmdWithADouble("/gate/application/SetNumberOfPrimariesPerRun", this);
  SetNumberOfPrimariesPerRunCmd2->SetGuidance("Set the number of primaries to generate per per run.");

  //LSLS
  ReadNumberOfPrimariesInAFileCmd = new G4UIcmdWithAString("/gate/application/readNumberOfPrimariesInAFile", this);
  ReadNumberOfPrimariesInAFileCmd->SetGuidance("Read the number of primaries per run in a file.");


  TimeStudyCmd = new G4UIcmdWithAString("/gate/application/enableTrackTimeStudy", this);
  TimeStudyCmd->SetGuidance("Activate the time measurement of tracks (Slow down the simulation).");
  TimeStudyCmd->SetParameterName("File name",false);

  TimeStudyForStepsCmd = new G4UIcmdWithAString("/gate/application/enableStepAndTrackTimeStudy", this);
  TimeStudyForStepsCmd->SetGuidance("Activate the time measurement of steps and tracks (Slow down the simulation).");
  TimeStudyForStepsCmd->SetParameterName("File name",false);
}
//-------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------
GateApplicationMgrMessenger::~GateApplicationMgrMessenger()
{
  delete GateApplicationDir;
  delete TimeSliceCmd;
  delete TimeStartCmd;
  delete TimeStopCmd;
  delete StartDAQCmd;
  delete StartCmd;
  delete StartDAQCompleteCmd;
  delete StartDAQClusterCmd;
  delete StopDAQCmd;
  delete PauseDAQCmd;
  delete VerboseCmd;
  //  delete ExitFlagCmd;
  delete DescribeCmd;
  delete NoOutputCmd;
  //delete EnableSuccessiveSourceMode;
  delete ReadTimeSlicesInAFileCmd;
  delete SetTotalNumberOfPrimariesCmd;
  delete SetNumberOfPrimariesPerRunCmd;
  delete SetNumberOfPrimariesPerRunCmd2;
  delete AddSliceCmd;
  delete TimeStudyCmd;
  delete TimeStudyForStepsCmd;

  //LSLS
  delete ReadNumberOfPrimariesInAFileCmd;

}
//-------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------
void GateApplicationMgrMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  //   G4cout << " GateApplicationMgrMessenger::SetNewValue " << newValue << Gateendl;

  GateApplicationMgr* appMgr = GateApplicationMgr::GetInstance();

  if( command == TimeSliceCmd ) {
    appMgr->SetTimeSlice(TimeSliceCmd->GetNewDoubleValue(newValue));
  }
  else   if( command == TimeStartCmd ) {
    appMgr->SetTimeStart(TimeStartCmd->GetNewDoubleValue(newValue));
  }
  else  if( command == TimeStopCmd ) {
    appMgr->SetTimeStop(TimeStopCmd->GetNewDoubleValue(newValue));
  }
  else  if( command == AddSliceCmd ) {
    appMgr->SetTimeInterval(AddSliceCmd->GetNewDoubleValue(newValue));
  }
  else  if( command == StartDAQCmd ) {
    appMgr->StartDAQ();
  }
  else  if( command == StartCmd ) {
    appMgr->StartDAQ();
  }
  else  if( command == StartDAQCompleteCmd ) {
    appMgr->StartDAQComplete(StartDAQCompleteCmd->GetNew3VectorValue(newValue));
  }
  else  if( command == StartDAQClusterCmd ) {
    appMgr->StartDAQCluster(StartDAQClusterCmd->GetNew3VectorValue(newValue));
  }
  else  if( command == StopDAQCmd ) {
    appMgr->StopDAQ();
  }
  else  if( command == PauseDAQCmd ) {
    appMgr->PauseDAQ();
  }
  else if( command == VerboseCmd ) {
    appMgr->SetVerboseLevel(VerboseCmd->GetNewIntValue(newValue));
  }
  //  else if( command == ExitFlagCmd ) {
  //    appMgr->SetExitFlag(ExitFlagCmd->GetNewBoolValue(newValue));
  //  }
  else  if( command == DescribeCmd ) {
    appMgr->Describe();
  }
  else if (command == NoOutputCmd) {
    appMgr->SetNoOutputMode();
  }
  //else if (command == EnableSuccessiveSourceMode) {
  //  appMgr->EnableSuccessiveSourceMode(true);
  //}
  else if (command == ReadTimeSlicesInAFileCmd) {
    appMgr->ReadTimeSlicesInAFile(newValue);
  }
  else if (command == SetTotalNumberOfPrimariesCmd) {
    double f=1.;
    GateSourceMgr * sourceMgr = GateSourceMgr::GetInstance();
    if (sourceMgr->GetNumberOfSources() == 1) {
      if(sourceMgr->GetSource(0)->GetName()=="PGS"){
        sourceMgr->GetSource(0)->Initialize();
        f=sourceMgr->GetSource(0)->GetSourceWeight();
        GateMessage("Run", 0, "Requested number of proton is " << newValue
                    << ". According to the PGS source, it's scaled with factor "
                    << f << "." << std::endl);
      }
    }
    appMgr->SetTotalNumberOfPrimaries(SetTotalNumberOfPrimariesCmd->GetNewDoubleValue(newValue)*f);
  }
  else if (command == SetNumberOfPrimariesPerRunCmd) {
    appMgr->SetNumberOfPrimariesPerRun(SetNumberOfPrimariesPerRunCmd->GetNewDoubleValue(newValue));
  }
  else if (command == SetNumberOfPrimariesPerRunCmd2) {
    appMgr->SetNumberOfPrimariesPerRun(SetNumberOfPrimariesPerRunCmd2->GetNewDoubleValue(newValue));
  }
  else if (command == ReadNumberOfPrimariesInAFileCmd) {
  appMgr->ReadNumberOfPrimariesInAFile(newValue);
  }
  else if (command == TimeStudyCmd) {
    appMgr->EnableTimeStudy(newValue);
  }
  else if (command == TimeStudyForStepsCmd) {
    appMgr->EnableTimeStudyForSteps(newValue);
  }
}
//-------------------------------------------------------------------------------------------------------------------
