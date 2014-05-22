/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEGENERICREPEATERMESSENGER_H
#define GATEGENERICREPEATERMESSENGER_H 1

#include "globals.hh"
#include "GateObjectRepeaterMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;

class GateGenericRepeater;

//-------------------------------------------------------------------------------------------------
/*! 
  \class GateGenericRepeaterMessenger
  \brief Messenger for a GateGenericRepeater
*/      
class GateGenericRepeaterMessenger: public GateObjectRepeaterMessenger
{
public:
  GateGenericRepeaterMessenger(GateGenericRepeater* itsRepeater);
  ~GateGenericRepeaterMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);
  
public:
  virtual inline GateGenericRepeater* GetGenericRepeater() { return (GateGenericRepeater*)GetObjectRepeater(); }
    
protected:
  G4UIcmdWithAString * mFileCmd;
  G4UIcmdWithABool   * mRelativeTransCmd; 
};
//-------------------------------------------------------------------------------------------------

#endif

