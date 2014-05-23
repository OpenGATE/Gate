/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVSourceMessenger_h
#define GateVSourceMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "GateMessenger.hh"

class GateVSource;

class GateClock;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithABool;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
//class GateUIcmdWithADoubleWithUnitAndInteger;

#include "GateUIcmdWithAVector.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateVSourceMessenger: public GateMessenger
{
public:
  GateVSourceMessenger(GateVSource* source);
  ~GateVSourceMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);
    
protected:
  GateVSource*                        m_source;
    
//    G4UIdirectory*                       GateSourceDir;
  G4UIcmdWithADoubleAndUnit*           ActivityCmd;
  G4UIcmdWithADoubleAndUnit*           StartTimeCmd;
  G4UIcmdWithAString*                  TypeCmd;
  G4UIcmdWithAnInteger*                DumpCmd;
  //G4UIcmdWithAnInteger*                NbrOfParticlesCmd;
  G4UIcmdWithABool*                    ForcedUnstableCmd;
  G4UIcmdWithADoubleAndUnit*           ForcedLifeTimeCmd;
  G4UIcmdWithABool*                    AccolinearityCmd;
  G4UIcmdWithADoubleAndUnit*           AccoValueCmd;
  G4UIcmdWithADoubleAndUnit*           ForcedHalfLifeCmd;
  G4UIcmdWithAnInteger*                VerboseCmd;
  //G4UIcmdWithADoubleAndUnit*           BeamTimeCmd;
  //G4UIcmdWithADouble*                  WeightCmd;
  G4UIcmdWithADouble*                  IntensityCmd;
  //G4UIcmdWithAString*                  TimeActivityCmd;
  //GateUIcmdWithADoubleWithUnitAndInteger* TimeParticleSliceCmd;
  G4UIcmdWithADoubleAndUnit*           setMinEnergycmd;
  G4UIcmdWithADoubleAndUnit*           setEnergyRangecmd;
  G4UIcmdWithAString*                  VisualizeCmd;
  G4UIcommand*                         useDefaultHalfLifeCmd;
};

#endif

