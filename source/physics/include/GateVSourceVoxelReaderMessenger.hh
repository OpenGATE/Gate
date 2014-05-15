/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVSourceVoxelReaderMessenger_h
#define GateVSourceVoxelReaderMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "GateMessenger.hh"

class GateVSourceVoxelReader;

class GateClock;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

#include "GateUIcmdWithAVector.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateVSourceVoxelReaderMessenger: public GateMessenger
{
public:
  GateVSourceVoxelReaderMessenger(GateVSourceVoxelReader* voxelReader);
  ~GateVSourceVoxelReaderMessenger();
  
  void SetNewValue(G4UIcommand* , G4String);
    
protected:
  GateVSourceVoxelReader*             m_voxelReader;
    
  G4UIcmdWith3VectorAndUnit*          PositionCmd;
  G4UIcmdWith3VectorAndUnit*          VoxelSizeCmd;

  GateUIcmdWithAVector<G4String>*     InsertTranslatorCmd;
  G4UIcmdWithoutParameter*            RemoveTranslatorCmd;
  G4UIcmdWithAnInteger*               VerboseCmd;
  G4UIcmdWithAString*              TimeActivTablesCmd; /* PY Descourt 08/09/2009 */
  G4UIcmdWithADoubleAndUnit*    SetTimeSamplingCmd;
};

#endif

