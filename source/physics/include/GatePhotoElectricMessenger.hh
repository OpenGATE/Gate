/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEPHOTOELECTRICPROCESSMESSENGER_HH
#define GATEPHOTOELECTRICPROCESSMESSENGER_HH


#include "GateEMStandardProcessMessenger.hh"
#include "GateVProcess.hh"


#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

class GatePhotoElectricMessenger:public GateEMStandardProcessMessenger
{
public:
  GatePhotoElectricMessenger(GateVProcess *pb);
  virtual ~GatePhotoElectricMessenger();

  virtual void BuildCommands(G4String base);
  virtual void SetNewValue(G4UIcommand*, G4String);

  bool GetIsAugerActivated(){return mAuger; }
  double GetLowEnergyElectronCut(){return mLowEnergyElectron;}
  double GetLowEnergyGammaCut(){return mLowEnergyGamma;}


protected:
  G4UIcmdWithABool * pActiveAugerCmd;
  G4UIcmdWithADoubleAndUnit *pSetLowEElectron;
  G4UIcmdWithADoubleAndUnit *pSetLowEGamma;

  bool mAuger;
  double mLowEnergyElectron;
  double mLowEnergyGamma;

};

#endif 
