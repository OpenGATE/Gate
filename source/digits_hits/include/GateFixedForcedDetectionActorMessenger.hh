/*----------------------
 GATE version name: gate_v6

 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

#ifndef GATEFIXEDFORCEDDECTECTIONACTORMESSENGER_HH
#define GATEFIXEDFORCEDDECTECTIONACTORMESSENGER_HH

#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"

#include "globals.hh"
#include "GateFixedForcedDetectionActor.hh"
#include "GateActorMessenger.hh"
#include "GateUIcmdWith2Vector.hh"

class GateFixedForcedDetectionActor;
class GateFixedForcedDetectionActorMessenger: public GateActorMessenger
  {
public:
  GateFixedForcedDetectionActorMessenger(GateFixedForcedDetectionActor* sensor);
  virtual ~GateFixedForcedDetectionActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateFixedForcedDetectionActor * pActor;
  G4UIcmdWithAString * pSetDetectorCmd;
  GateUIcmdWith2Vector * pSetDetectorResolCmd;
  GateUIcmdWith2Vector * pSetBinningFactorCmd;
  G4UIcmdWithAString * pSetGeometryFilenameCmd;
  G4UIcmdWithAString * pSetPrimaryFilenameCmd;
  G4UIcmdWithAString * pSetMaterialMuFilenameCmd;
  G4UIcmdWithAString * pSetAttenuationFilenameCmd;
  G4UIcmdWithAString * pSetMaterialDeltaFilenameCmd;
  G4UIcmdWithAString * pSetFresnelFilenameCmd;
  G4UIcmdWithAString * pSetResponseDetectorFilenameCmd;
  G4UIcmdWithAString * pSetFlatFieldFilenameCmd;
  G4UIcmdWithAString * pSetComptonFilenameCmd;
  G4UIcmdWithAString * pSetRayleighFilenameCmd;
  G4UIcmdWithAString * pSetFluorescenceFilenameCmd;
  G4UIcmdWithAString * pSetIsotropicPrimaryFilenameCmd;
  G4UIcmdWithAString * pSetSourceTypeCmd;
  G4UIcmdWithAString * pSetGeneratePhotonsCmd;
  G4UIcmdWithAString * pSetARFCmd;
  G4UIcmdWithAString * pSetSecondaryFilenameCmd;
  G4UIcmdWithABool * pEnableSecondarySquaredCmd;
  G4UIcmdWithABool * pEnableSecondaryUncertaintyCmd;
  G4UIcmdWithAString * pSetTotalFilenameCmd;
  G4UIcmdWithAString * pSetPhaseSpaceFilenameCmd;
  G4UIcmdWithAString * pSetInputRTKGeometryFilenameCmd;
  G4UIcmdWithAnInteger * pSetNoisePrimaryCmd;
  G4UIcmdWithADoubleAndUnit * pEnergyResolvedBinSizeCmd;
  };

#endif /* end #define GATEFIXEDFORCEDDECTECTIONACTORMESSENGER_HH*/
#endif // GATE_USE_RTK
