/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/



#ifndef GATEPHYSICSLISTMESSENGER_HH
#define GATEPHYSICSLISTMESSENGER_HH

#include "globals.hh"

#include "G4UImessenger.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"
#include "GateUIcmdWithAStringAndAnInteger.hh"
#include "GateUIcmdWith2String.hh"
#include "GateMaterialMuHandler.hh"

class GatePhysicsList;

class GatePhysicsListMessenger:public G4UImessenger
{
public:
  GatePhysicsListMessenger(GatePhysicsList * pl);
  ~GatePhysicsListMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

  G4double ScaleValue(G4double value,G4String unit);

protected:
  GatePhysicsList * pPhylist;
  GateUIcmdWith2String * pRemove;
  GateUIcmdWith2String * pAdd;
  GateUIcmdWith2String * pAddProcessMixed;

  GateUIcmdWith2String * pList;
  G4UIcmdWithoutParameter * pInit;
  G4UIcmdWithAString * pPrint;

  G4UIcmdWithAString * gammaCutCmd;
  G4UIcmdWithAString * electronCutCmd;
  G4UIcmdWithAString * positronCutCmd;
  G4UIcmdWithAString * protonCutCmd;

  G4UIcmdWithAString * pMaxStepSizeCmd;
  G4UIcmdWithAString * pMaxTrackLengthCmd;
  G4UIcmdWithAString * pMaxToFCmd;
  G4UIcmdWithAString * pMinKineticEnergyCmd;
  G4UIcmdWithAString * pMinRemainingRangeCmd;

  G4UIcmdWithAString * pActivateStepLimiterCmd;
  G4UIcmdWithAString * pActivateSpecialCutsCmd;

  G4UIcmdWithoutParameter * pCutInMaterial;

  G4UIcmdWithAnInteger * pSetDEDXBinning;
  G4UIcmdWithAnInteger * pSetLambdaBinning;
  G4UIcmdWithADoubleAndUnit * pSetEMin;
  G4UIcmdWithADoubleAndUnit * pSetEMax;
  G4UIcmdWithABool * pSetSplineFlag;
#if G4VERSION_MAJOR >= 10 && G4VERSION_MINOR >= 5
  G4UIcmdWithABool * pSetUseICRU90DataFlag;
#endif

  // Mu Handler tools
  G4UIcmdWithAString * pMuHandlerSetDatabase;
  G4UIcmdWithADoubleAndUnit * pMuHandlerSetEMin;
  G4UIcmdWithADoubleAndUnit * pMuHandlerSetEMax;
  G4UIcmdWithAnInteger * pMuHandlerSetENumber;
  G4UIcmdWithADoubleAndUnit * pMuHandlerSetAtomicShellEMin;
  G4UIcmdWithADoubleAndUnit * pMuHandlerSetAtomicShellTolerance;
  G4UIcmdWithADouble * pMuHandlerSetPrecision;

  G4UIcommand * pAddAtomDeexcitation;
  G4UIcmdWithAString * pAddPhysicsList;
  G4UIcmdWithAString * pAddPhysicsListMixed;
  G4UIcmdWithABool * pConstructProcessMixed;

  G4UIcmdWithADoubleAndUnit * pEnergyRangeMinLimitCmd;

private:
  int nInit;
  int nEMStdOpt;
  GateMaterialMuHandler *nMuHandler;
};

#endif /* end #define GATEPHYSICSLISTMESSENGER_HH */
