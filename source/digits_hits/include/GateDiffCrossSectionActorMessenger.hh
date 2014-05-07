/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateDiffCrossSectionActorMessenger
  \author edward.romero@creatis.insa-lyon.fr
*/

#ifndef GATEDiffCrossSectionACTORMESSENGER_HH
#define GATEDiffCrossSectionACTORMESSENGER_HH

#include "GateActorMessenger.hh"
//#include "G4UIcmdWithADoubleAndUnit.hh"
class G4UIcmdWithADoubleAndUnit;
class GateDiffCrossSectionActor;
class GateDiffCrossSectionActorMessenger : public GateActorMessenger
{
public:
  GateDiffCrossSectionActorMessenger(GateDiffCrossSectionActor* sensor);
  virtual ~GateDiffCrossSectionActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateDiffCrossSectionActor * pDiffCrossSectionActor;

  G4UIcmdWithADoubleAndUnit* pSetEnergyCmd;
  G4UIcmdWithADoubleAndUnit* pSetAngleCmd;
  G4UIcmdWithAString* pReadListEnergyCmd;
  G4UIcmdWithAString* pReadListAngleCmd;
  G4UIcmdWithAString * pSetMaterialCmd;
  G4UIcmdWithAString * pReadMaterialListCmd;

};

#endif
