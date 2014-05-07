/*----------------------
   OpenGATE Collaboration

   Giovanni Santin <giovanni.santin@cern.ch>
   Daniel Strul <daniel.strul@iphe.unil.ch>

   Copyright (C) 2002 UNIL/IPHE, CH-1015 Lausanne

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \file GateSteppingActionMessenger.hh

  $Log: GateSteppingActionMessenger.hh,v $
  Revision 1.2  2002/08/11 15:33:24  dstrul
  Cosmetic cleanup: standardized file comments for cleaner doxygen output


  \brief Class GateSteppingActionMessenger
  \brief By Giovanni.Santin@cern.ch (Apr 7, 2002)
  \brief $Id: GateSteppingActionMessenger.hh,v 1.2 2002/08/11 15:33:24 dstrul Exp $
*/


#ifndef GateSteppingActionMessenger_h
#define GateSteppingActionMessenger_h 1

class GateSteppingAction;
class G4UIdirectory;
class G4UIcmdWithAnInteger;
class G4UIcmdWithABool;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithAString;

#include "G4UImessenger.hh"
#include "globals.hh"

class GateSteppingActionMessenger: public G4UImessenger
{
  public:
    GateSteppingActionMessenger(GateSteppingAction* msa);
    ~GateSteppingActionMessenger();

  public:
    void SetNewValue(G4UIcommand * command,G4String newValues);

  private:
    GateSteppingAction * myAction;

private: //commands
  G4UIdirectory*         GateSteppingDir;
  G4UIcmdWithAnInteger*  drawTrajectoryLevelCmd;
  G4UIcmdWithAnInteger*  VerboseCmd;
  G4UIcmdWithAString*    SetModeCmd;
  G4UIcmdWithAString*    PolicyCmd;
  G4UIcmdWithAString*    GetTxtCmd;
  G4UIcmdWithAnInteger*  SetFilesCmd;
  //G4UIcmdWithAnInteger*  SetPhFilesCmd;
  //G4UIcmdWithAnInteger*  SetRSFilesCmd;
  G4UIcmdWithADoubleAndUnit* setEnergyTcmd;

};

#endif
