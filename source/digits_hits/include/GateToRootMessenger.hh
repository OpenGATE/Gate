/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateToRootMessenger_h
#define GateToRootMessenger_h 1

#ifdef G4ANALYSIS_USE_ROOT

#include "GateOutputModuleMessenger.hh"
#include "GateToRoot.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateToRootMessenger: public GateOutputModuleMessenger
{
  public:
    GateToRootMessenger(GateToRoot* gateToRoot);
   ~GateToRootMessenger();
    
    void CreateNewOutputChannelCommand(GateToRoot::VOutputChannel* anOutputChannel);

    void SetNewValue(G4UIcommand*, G4String);
    
    G4bool IsAnOutputChannelCmd(G4UIcommand* command);
    void ExecuteOutputChannelCmd(G4UIcommand* command,G4String newValue);

  protected:
    GateToRoot*             m_gateToRoot;
    
    G4UIcmdWithoutParameter* ResetCmd;

    G4UIcmdWithABool*        RootHitCmd;
    G4UIcmdWithABool*        RootNtupleCmd;
		G4UIcmdWithABool*        RootRecordCmd;
    G4UIcmdWithABool*        SaveRndmCmd;
    G4UIcmdWithAString*      SetFileNameCmd;

    std::vector<G4UIcmdWithABool*>  		 OutputChannelCmdList;
    std::vector<GateToRoot::VOutputChannel*>  m_outputChannelList;
};

#endif
#endif

