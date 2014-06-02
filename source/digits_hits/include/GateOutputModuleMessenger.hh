/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateOutputModuleMessenger_h
#define GateOutputModuleMessenger_h 1

#include "GateMessenger.hh"

class GateVOutputModule;

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateOutputModuleMessenger: public GateMessenger
{
public:
  GateOutputModuleMessenger(GateVOutputModule* outputModule);
  virtual ~GateOutputModuleMessenger();

  virtual void SetNewValue(G4UIcommand*, G4String);

public:
  inline G4UIcmdWithoutParameter* GetDescribeCmd() {return DescribeCmd;}
  inline G4UIcmdWithAnInteger*    GetVerboseCmd()  {return VerboseCmd;}
  inline G4UIcmdWithoutParameter* GetEnableCmd()   {return EnableCmd;}
  inline G4UIcmdWithoutParameter* GetDisableCmd()  {return DisableCmd;}

private:
  GateVOutputModule*                       m_outputModule;

  G4UIcmdWithoutParameter*                 DescribeCmd;
  G4UIcmdWithAnInteger*                    VerboseCmd;
  G4UIcmdWithoutParameter*                 EnableCmd;    //!< The UI command "enable"
  G4UIcmdWithoutParameter*                 DisableCmd;   //!< The UI command "disable"
};

#endif
