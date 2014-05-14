/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEGENERICMOVEMESSENGERMESSENGER_H
#define GATEGENERICMOVEMESSENGERMESSENGER_H 1

#include "globals.hh"
#include "GateObjectRepeaterMessenger.hh"
#include "GateGenericMove.hh"

class G4UIdirectory;
class G4UIcmdWithAString;

//-------------------------------------------------------------------------------------------------
class GateGenericMoveMessenger: public GateObjectRepeaterMessenger
{
public:
  GateGenericMoveMessenger(GateGenericMove* itsMove);
  ~GateGenericMoveMessenger();

  void SetNewValue(G4UIcommand*, G4String);

  virtual inline GateGenericMove* GetGenericMove() { return (GateGenericMove*)GetObjectRepeater(); }
  
protected:
  G4UIcmdWithAString * mFilenameCmd;
  G4UIcmdWithABool   * mRelativeTransCmd;
};
//-------------------------------------------------------------------------------------------------

#endif
