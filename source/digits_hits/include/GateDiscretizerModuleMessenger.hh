/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateDiscretizerModule.cc for more detals
  */


/*! \class  GateDiscretizerModuleMessenger
    \brief  Messenger for the GateDiscretizerModule

    - GateDiscretizerModule - by marc.granado@universite-paris-saclay.fr

    \sa GateDiscretizerModule, GateDiscretizerModuleMessenger
*/


#ifndef GateDiscretizerModuleMessenger_h
#define GateDiscretizerModuleMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateDiscretizerModule;
class G4UIcmdWithAString;

class GateDiscretizerModuleMessenger : public GateClockDependentMessenger
{
public:
  
  GateDiscretizerModuleMessenger(GateDiscretizerModule*);
  ~GateDiscretizerModuleMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateDiscretizerModule* m_DiscretizerModule;

  G4UIcmdWithAString*	nameAxisCmd;

  G4UIcmdWithADouble*  	resCmd;
  G4UIcmdWithADouble*   resolutionXCmd;
  G4UIcmdWithADouble*   resolutionYCmd;
  G4UIcmdWithADouble*   resolutionZCmd;
};

#endif








