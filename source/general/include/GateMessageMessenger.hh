/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateMessageMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEMESSAGEMessenger_h
#define GATEMESSAGEMessenger_h


#include "G4UImessenger.hh"
//#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithAString.hh"

class GateMessageManager;

//-----------------------------------------------------------------------------
class GateMessageMessenger : public G4UImessenger
{
public:
  /// Ctor
  GateMessageMessenger(G4String base, GateMessageManager* man);
  /// Dtor
  ~GateMessageMessenger();

  /// Command processing callback
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateMessageManager * pMessageManager;
  G4UIcmdWithAString* pVerboseCmd;

};
// EO class GateManager
//-----------------------------------------------------------------------------

#endif
