/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*  Optical Photons: V. Cuplov -  2012
         - Revision 2012/09/17  /gate/output/root/setRootOpticalFlag functionality added.
           Set the flag for Optical ROOT output.
*/


#ifndef GateToRootMessenger_h
#define GateToRootMessenger_h 1

#include "GateConfiguration.h"

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
class G4UIcommand;

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
    G4UIcmdWithABool*        RootCCCmd;
    G4UIcmdWithABool*        RootCCSourceParentIDSpecificationCmd;
    G4UIcmdWithABool*        RootHitCmd;
    G4UIcmdWithABool*        RootSinglesCmd;
    G4UIcmdWithABool*        RootCoincidencesCmd;

    G4UIcmdWithABool*        RootNtupleCmd;
    G4UIcmdWithABool*        RootOpticalCmd;
    G4UIcmdWithABool*        RootRecordCmd;
    G4UIcmdWithABool*        SaveRndmCmd;
    G4UIcmdWithAString*      SetFileNameCmd;

    G4UIcommand*      CoincidenceMaskCmd;
	G4int m_coincidenceMaskLength;

    G4UIcommand*      SingleMaskCmd;
	G4int m_singleMaskLength;

    std::vector<G4UIcmdWithABool*>  		 OutputChannelCmdList;
  public:
    std::vector<GateToRoot::VOutputChannel*>  m_outputChannelList;
};

#endif
#endif
