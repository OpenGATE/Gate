/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateGeometryVoxelImageReaderMessenger.hh"
#include "GateGeometryVoxelImageReader.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateGeometryVoxelImageReaderMessenger::GateGeometryVoxelImageReaderMessenger(GateGeometryVoxelImageReader* voxelReader)
  : GateVGeometryVoxelReaderMessenger((GateVGeometryVoxelReader*)voxelReader)
{ 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateGeometryVoxelImageReaderMessenger::~GateGeometryVoxelImageReaderMessenger()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateGeometryVoxelImageReaderMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  GateVGeometryVoxelReaderMessenger::SetNewValue(command,newValue);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....



