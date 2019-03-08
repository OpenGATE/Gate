/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GateApplicationMgrMessenger_h
#define GateApplicationMgrMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

class GateApplicationMgr;
class G4UIdirectory;
class G4UIcmdWithABool;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateApplicationMgrMessenger: public G4UImessenger
{
public:
  GateApplicationMgrMessenger();
  ~GateApplicationMgrMessenger();

  void SetNewValue(G4UIcommand*, G4String);

private:
  G4UIdirectory*           GateApplicationDir;

  G4UIcmdWithADoubleAndUnit* TimeSliceCmd;
  G4UIcmdWithADoubleAndUnit* TimeStartCmd;
  G4UIcmdWithADoubleAndUnit* TimeStopCmd;
  G4UIcmdWithADoubleAndUnit* AddSliceCmd;
  G4UIcmdWithoutParameter*   StartDAQCmd;
  G4UIcmdWithoutParameter*   StartCmd;
  G4UIcmdWith3VectorAndUnit* StartDAQCompleteCmd;
  //dk cluster
  G4UIcmdWith3VectorAndUnit* StartDAQClusterCmd;
  //dk cluster end
  G4UIcmdWithoutParameter*   StopDAQCmd;
  G4UIcmdWithoutParameter*   PauseDAQCmd;
  G4UIcmdWithAnInteger*      VerboseCmd;
//  G4UIcmdWithABool*          ExitFlagCmd;
  G4UIcmdWithoutParameter*   DescribeCmd;

  G4UIcmdWithoutParameter * NoOutputCmd;
  G4UIcmdWithAString * TimeStudyCmd;
  G4UIcmdWithAString * TimeStudyForStepsCmd;
  //G4UIcmdWithoutParameter * EnableSuccessiveSourceMode;
  G4UIcmdWithAString *      ReadTimeSlicesInAFileCmd;
  G4UIcmdWithADouble *      SetTotalNumberOfPrimariesCmd;
  G4UIcmdWithADouble *      SetNumberOfPrimariesPerRunCmd;
  G4UIcmdWithADouble *      SetNumberOfPrimariesPerRunCmd2;

//LSLS
  G4UIcmdWithAString *      ReadNumberOfPrimariesInAFileCmd;

};

#endif
