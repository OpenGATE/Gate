/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateNanoActorMessenger
  \author 
*/

#ifndef GATENANOACTORMESSENGER_HH
#define GATENANOACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "GateImageActorMessenger.hh"
#include "G4UIcmdWithADouble.hh"

class GateNanoActor;
class GateNanoActorMessenger : public GateImageActorMessenger
{
public:
  GateNanoActorMessenger(GateNanoActor* sensor);
  virtual ~GateNanoActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateNanoActor * pNanoActor;

  G4UIcmdWithABool * pEnableNanoAbsorptionCmd;

  G4UIcmdWithADouble* pSigmaCmd;
  G4UIcmdWithADouble* pTimeCmd;
  G4UIcmdWithADouble* pDiffusivityCmd;


};

#endif /* end #define GATENANOACTORMESSENGER_HH*/
