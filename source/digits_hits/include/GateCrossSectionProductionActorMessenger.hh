/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateDoseActorMessenger
  \author claire.vanngocty@gmail.com
*/

#ifndef GATECROSSSECTIONPRODUCTIONACTORMESSENGER_HH
#define GATECROSSSECTIONPRODUCTIONACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "GateImageActorMessenger.hh"

class GateCrossSectionProductionActor;
class GateCrossSectionProductionActorMessenger : public GateImageActorMessenger
{
public:
  GateCrossSectionProductionActorMessenger(GateCrossSectionProductionActor* sensor);
  virtual ~GateCrossSectionProductionActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateCrossSectionProductionActor * pCrossSectionProductionActor;
  G4UIcmdWithABool * pO15Cmd;
  G4UIcmdWithABool * pC11Cmd;
  G4UIcmdWithAString * pC11FilenameCmd;
};

#endif
