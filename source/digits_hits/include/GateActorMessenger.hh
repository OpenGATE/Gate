/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateActorMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
          david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEACTORMESSENGER_HH
#define GATEACTORMESSENGER_HH

#include "globals.hh"
#include "G4UImessenger.hh"

class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWithABool;

#include "GateActorManager.hh"

class GateVActor;

class GateActorMessenger : public  G4UImessenger
{
public:
  GateActorMessenger(GateVActor* sensor);
  virtual ~GateActorMessenger();

  virtual void BuildCommands(G4String base);
  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  GateVActor * pActor;

  G4UIcmdWithAString*   pSetFileNameCmd;
  G4UIcmdWithAString*   pSetVolumeNameCmd;
  G4UIcmdWithAnInteger* pSaveEveryNEventsCmd;
  G4UIcmdWithAnInteger* pSaveEveryNSecondsCmd;
  G4UIcmdWithABool *    pSetOverWriteFilesFlagCmd;
  G4UIcmdWithABool *    pSetResetDataAtEachRunFlagCmd;
  G4UIcmdWithAString *  pAddFilterCmd;

  G4String baseName;
};

#endif /* end #define GATEACTORMESSENGER_HH*/
