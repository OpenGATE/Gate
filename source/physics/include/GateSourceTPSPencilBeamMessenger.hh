/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT
#ifndef GateSourceTPSPencilBeamMessenger_h
#define GateSourceTPSPencilBeamMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "GateMessenger.hh"
#include "GateVSourceMessenger.hh"

class GateSourceTPSPencilBeam;
class G4ParticleTable;
class G4UIcommand;
class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithABool;
class G4UIcmdWithoutParameter;
class GateVSource;


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateSourceTPSPencilBeamMessenger: public GateVSourceMessenger
{
  public:
    GateSourceTPSPencilBeamMessenger(GateSourceTPSPencilBeam* source);
    ~GateSourceTPSPencilBeamMessenger();
    void SetNewValue(G4UIcommand*, G4String);

  private:
    GateSourceTPSPencilBeam * pSourceTPSPencilBeam;

    //Particle Type
    G4UIcmdWithAString * pParticleTypeCmd;
    //Configuration of tests
    G4UIcmdWithABool* pTestCmd;
    //Treatment Plan file
    G4UIcmdWithAString * pPlanCmd;
    //FlatGenerationFlag
    G4UIcmdWithABool * pFlatGeneFlagCmd;
    //Not allowed fieldID
    G4UIcmdWithAnInteger * pNotAllowedFieldCmd;
    //Source description file
    G4UIcmdWithAString * pSourceFileCmd;
    //Configuration of spot intensity
    G4UIcmdWithABool * pSpotIntensityCmd;
    //Convergent or divergent beam model
    G4UIcmdWithABool* pDivergenceCmd;
    //Selection of one layer
    G4UIcmdWithAnInteger * pSelectLayerIDCmd;
};

#endif

#endif
