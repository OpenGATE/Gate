/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateDummyDigitizerModule.cc for more detals
  */


/*! \class  GateDummyDigitizerModuleMessenger
    \brief  Messenger for the GateDummyDigitizerModule

    - GateDummyDigitizerModule - by name.surname@email.com

    \sa GateDummyDigitizerModule, GateDummyDigitizerModuleMessenger
*/


#ifndef GateDummyDigitizerModuleMessenger_h
#define GateDummyDigitizerModuleMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateDummyDigitizerModule;
class G4UIcmdWithAString;

class GateDummyDigitizerModuleMessenger : public GateClockDependentMessenger
{
public:
  
  GateDummyDigitizerModuleMessenger(GateDummyDigitizerModule*);
  ~GateDummyDigitizerModuleMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateDummyDigitizerModule* m_DummyDigitizerModule;
  G4UIcmdWithAString          *dummyCmd;


};

#endif








