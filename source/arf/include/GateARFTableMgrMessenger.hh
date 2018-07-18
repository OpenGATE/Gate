/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

#ifndef GateARFTableMgrMessenger_h
#define GateARFTableMgrMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

class GateARFTableMgr;
class GateClock;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

#include "GateUIcmdWithAVector.hh"

class GateARFTableMgrMessenger: public G4UImessenger
  {
public:
  GateARFTableMgrMessenger(G4String aName, GateARFTableMgr*);
  ~GateARFTableMgrMessenger();
  void SetNewValue(G4UIcommand*, G4String);

private:
  GateARFTableMgr* mArfTableMgr;
  G4UIdirectory* mGateARFTableDir;
  G4UIcmdWithAString* mCptTableEWCmd;
  G4UIcmdWithoutParameter* mListARFTableCmd;
  G4UIcmdWithAnInteger* mVerboseCmd;
  G4UIcmdWithADouble* mSetEResocmd;
  G4UIcmdWithADoubleAndUnit* mSetERefcmd;
  G4UIcmdWithADoubleAndUnit* mSetEThreshHoldcmd;
  G4UIcmdWithADoubleAndUnit* mSetEUpHoldcmd;
  G4UIcmdWithAString* mSaveToBinaryFileCmd;
  G4UIcmdWithAnInteger* mSetNBinsCmd;
  G4UIcmdWithAString* mLoadFromBinaryFileCmd;
  G4UIcmdWithADoubleAndUnit* mSetDistancecmd;
  };

#endif
