/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*! \class  GatePileupMessenger
    \brief  Messenger for the GatePileup
    \sa GatePileup, GatePileupMessenger
*/


#ifndef GatePileupMessenger_h
#define GatePileupMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GatePileup;
class G4UIcmdWithAString;
class G4UIdirectory;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;

class GatePileupMessenger : public GateClockDependentMessenger
{
public:
  
	GatePileupMessenger(GatePileup* itsPileup);
	virtual~GatePileupMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GatePileup* m_Pileup;
  G4UIcmdWithAnInteger*      SetDepthCmd;
  G4UIcmdWithADoubleAndUnit* SetPileupCmd;
  G4UIcmdWithAString*        SetNewVolCmd;

};

#endif








