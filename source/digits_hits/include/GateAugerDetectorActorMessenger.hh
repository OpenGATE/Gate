#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

/*
  \class  GateAugerDetectorActorMessenger
  \author pierre.gueth@creatis.insa-lyon.fr
*/

#ifndef GATEAUGERDETECTORACTORMESSENGER_HH
#define GATEAUGERDETECTORACTORMESSENGER_HH

#include <G4UIcmdWithAnInteger.hh>
#include <G4UIcmdWithADoubleAndUnit.hh>
#include <G4UIcmdWith3Vector.hh>

#include "GateActorMessenger.hh"

class GateAugerDetectorActor;

class GateAugerDetectorActorMessenger : public GateActorMessenger
{
public:

  GateAugerDetectorActorMessenger(GateAugerDetectorActor * v);
  virtual ~GateAugerDetectorActorMessenger();
  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  void BuildCommands(G4String base);

  /// Associated sensor
  GateAugerDetectorActor * pActor;

  /// Command objects
  //G4UIcmdWithAnInteger * pNBinsCmd;
  G4UIcmdWithADoubleAndUnit* pMinTOFCmd;
  G4UIcmdWithADoubleAndUnit* pMaxTOFCmd;
  G4UIcmdWithADoubleAndUnit* pMinEdepCmd;
  G4UIcmdWithADoubleAndUnit* pMaxEdepCmd;
  G4UIcmdWith3Vector* pProfileDirectionCmd;
  G4UIcmdWithADoubleAndUnit* pMinProfileCmd;
  G4UIcmdWithADoubleAndUnit* pMaxProfileCmd;
  G4UIcmdWithAnInteger* pSizeProfileCmd;
  G4UIcmdWithADoubleAndUnit* pProfileNoiseFWHMCmd;
};

#endif
#endif
