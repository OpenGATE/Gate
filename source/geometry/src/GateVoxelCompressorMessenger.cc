/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVoxelCompressorMessenger.hh"
#include "GateVoxelCompressor.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"


GateVoxelCompressorMessenger::GateVoxelCompressorMessenger(GateVoxelCompressor *itsInserter)
  :GateMessenger(itsInserter->GetObjectName()+"/compression"),
   m_inserter(itsInserter)
{ 
 
  // G4cout << "GateVoxelCompressorMessenger::GateVoxelCompressorMessenger - Entered " << GetDirectoryName() << G4endl; 

  GetDirectory()->SetGuidance("Controls phantom compression.");

  G4String cmdName;

  // cmdName = G4String("/gate/") + itsInserter->GetObjectName()+ "/excludeList"; 
  cmdName = GetDirectoryName()+ "excludeList"; 
  MakeExclusionListCmd = new G4UIcmdWithAString(cmdName,this);
  MakeExclusionListCmd->SetGuidance("lists materials to be excluded from compression");

 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateVoxelCompressorMessenger::~GateVoxelCompressorMessenger()
{
   delete MakeExclusionListCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateVoxelCompressorMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
 if ( command ==  MakeExclusionListCmd )
    { m_inserter->MakeExclusionList(newValue); }
  else
    GateMessenger::SetNewValue(command,newValue);

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
