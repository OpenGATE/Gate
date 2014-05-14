/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePrimaryGeneratorMessenger_h
#define GatePrimaryGeneratorMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "GatePrimaryGeneratorAction.hh"

class GateClock;
class GatePrimaryGeneratorAction;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//-------------------------------------------------------------------------------------------------
class GatePrimaryGeneratorMessenger: public G4UImessenger
{
public:
  GatePrimaryGeneratorMessenger(GatePrimaryGeneratorAction*);
  ~GatePrimaryGeneratorMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);
    
private:
  GatePrimaryGeneratorAction* m_primaryGenerator;
    
  G4UIdirectory*             GateGeneratorDir;
  G4UIcmdWithAnInteger*      VerboseCmd;
  G4UIcmdWithoutParameter*   GPSCmd;
};
//-------------------------------------------------------------------------------------------------

#endif

