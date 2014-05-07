/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateAnalysisMessenger_h
#define GateAnalysisMessenger_h 1

#include "GateOutputModuleMessenger.hh"

class GateAnalysis;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateAnalysisMessenger: public GateOutputModuleMessenger
{
  public:
    GateAnalysisMessenger(GateAnalysis* gateAnalysis);
   ~GateAnalysisMessenger();

    virtual void SetNewValue(G4UIcommand*, G4String);

  protected:
    GateAnalysis*            m_gateAnalysis;
    G4UIcmdWithAString*      SetSeptalVolumeNameCmd; // HDS : Tells in which volume to record septal penetration
    G4UIcmdWithABool*        RecordSeptalCmd; // HDS: Flag to activate the recording of septal penetration in root trees
};

#endif
