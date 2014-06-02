/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEGENERICREPEATERMOVEMESSENGER_H
#define GATEGENERICREPEATERMOVEMESSENGER_H 1

#include "globals.hh"
#include "GateObjectRepeaterMessenger.hh"
#include "GateGenericRepeaterMove.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;

//-------------------------------------------------------------------------------------------------
class GateGenericRepeaterMoveMessenger: public GateObjectRepeaterMessenger
{
public:
  GateGenericRepeaterMoveMessenger(GateGenericRepeaterMove* itsMove);
  ~GateGenericRepeaterMoveMessenger();

  void SetNewValue(G4UIcommand*, G4String);

  virtual inline GateGenericRepeaterMove* GetGenericRepeaterMove() { return (GateGenericRepeaterMove*)GetObjectRepeater(); }
  
protected:
  G4UIcmdWithAString * mFilenameCmd;
  G4UIcmdWithABool   * mRelativeTransCmd;
};
//-------------------------------------------------------------------------------------------------

#endif
