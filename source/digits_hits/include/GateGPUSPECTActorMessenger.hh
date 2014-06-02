/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  GateGPUSPECTActorMessenger
*/

#ifndef GATEGPUSPECTACTORMESSENGER_HH
#define GATEGPUSPECTACTORMESSENGER_HH


#include "GateImageActorMessenger.hh"

class G4UIcmdWithAnInteger;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWith3Vector;

class GateGPUSPECTActor;
class GateGPUSPECTActorMessenger : public GateActorMessenger
{
public:
  GateGPUSPECTActorMessenger(GateGPUSPECTActor* sensor);
  virtual ~GateGPUSPECTActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateGPUSPECTActor * pSPECTActor;
  G4UIcmdWithAnInteger * pSetGPUDeviceIDCmd;
  G4UIcmdWithAnInteger * pSetGPUBufferCmd;
  G4UIcmdWithADoubleAndUnit * pSetHoleHexaHeightCmd;
  G4UIcmdWithADoubleAndUnit * pSetHoleHexaRadiusCmd;
  G4UIcmdWith3Vector * pSetHoleHexaRotAxisCmd;
  G4UIcmdWithADoubleAndUnit * pSetHoleHexaRotAngleCmd;
  G4UIcmdWithAString * pSetHoleHexaMaterialCmd;
  G4UIcmdWithAnInteger * pSetCubArrayRepNumXCmd;
  G4UIcmdWithAnInteger * pSetCubArrayRepNumYCmd;
  G4UIcmdWithAnInteger * pSetCubArrayRepNumZCmd;
  G4UIcmdWith3VectorAndUnit * pSetCubArrayRepVecCmd;
  G4UIcmdWithAnInteger * pSetLinearRepNumCmd;
  G4UIcmdWith3VectorAndUnit * pSetLinearRepVecCmd;  
  
};

#endif /* end #define GATEGPUSPECTACTOR_HH*/
