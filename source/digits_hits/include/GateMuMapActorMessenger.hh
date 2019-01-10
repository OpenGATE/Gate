/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateMuMapActorMessenger
  \author gsizeg@gmail.com
*/

#ifndef GATEMUMAPACTORMESSENGER_HH
#define GATEMUMAPACTORMESSENGER_HH

#include "GateImageActorMessenger.hh"

class G4UIcmdWithADoubleAndUnit;
class GateMuMapActor;

class GateMuMapActorMessenger : public GateImageActorMessenger
{
public:
  GateMuMapActorMessenger(GateMuMapActor* sensor);
  virtual ~GateMuMapActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateMuMapActor * pMuMapActor;

  G4UIcmdWithADoubleAndUnit* pSetEnergyCmd;
  G4UIcmdWithADoubleAndUnit* pSetMuUnitCmd;
};

#endif /* end #define GATEMUMAPACTORMESSENGER_HH*/
