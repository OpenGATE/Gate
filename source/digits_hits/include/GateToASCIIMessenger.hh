/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateToASCIIMessenger_h
#define GateToASCIIMessenger_h 1

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_FILE

#include "GateOutputModuleMessenger.hh"
#include <vector>

#include "GateToASCII.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateToASCIIMessenger: public GateOutputModuleMessenger
{
  public:
    GateToASCIIMessenger(GateToASCII* gateToASCII);
   ~GateToASCIIMessenger();

    void CreateNewOutputChannelCommand(GateToASCII::VOutputChannel* anOutputChannel);

    void SetNewValue(G4UIcommand*, G4String);

    G4bool IsAnOutputChannelCmd(G4UIcommand* command);
    void ExecuteOutputChannelCmd(G4UIcommand* command,G4String newValue);

  protected:
    GateToASCII*             			 m_gateToASCII;

    G4UIcmdWithoutParameter*          		 ResetCmd;
    G4UIcmdWithABool*                 		 OutFileHitsCmd;
    G4UIcmdWithABool*                 		 OutFileSinglesCmd;
    G4UIcmdWithABool*                 		 OutFileVoxelCmd;
    G4UIcmdWithAString*               		 SetFileNameCmd;

    std::vector<G4UIcmdWithABool*>  		 OutputChannelCmdList;
    std::vector<GateToASCII::VOutputChannel*>  m_outputChannelList;

    G4UIcommand* CoincidenceMaskCmd;
    G4int m_coincidenceMaskLength;

    G4UIcommand* SingleMaskCmd;
    G4int m_singleMaskLength;

    G4UIcmdWithAnInteger*                 	 SetOutFileSizeLimitCmd;

};

#endif
#endif
