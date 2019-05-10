/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GateSourceTPSPencilBeamMessenger_h
#define GateSourceTPSPencilBeamMessenger_h 1

#include "GateConfiguration.h"
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

//----------------------------------------------------------------------------------------
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
    //Particle Properties If GenericIon
    G4UIcommand * pIonCmd;
    //Set the test Flag for debugging (verbosity)
    G4UIcmdWithABool* pTestCmd;
    //Configuration of vertex generation method (random by default)
    G4UIcmdWithABool* pSortedSpotGenerationCmd;
    //Configuration of absolute/relative energy spread specification (relative by default)
    G4UIcmdWithABool* pSigmaEnergyInMeVCmd;
    //Treatment Plan file ("plan description file")
    G4UIcmdWithAString * pPlanCmd;
    //FlatGenerationFlag
    G4UIcmdWithABool * pFlatGeneFlagCmd;
    //Not allowed fieldID (all fields allowed by default)
    G4UIcmdWithAnInteger * pNotAllowedFieldCmd;
    //Allowed fieldID
    G4UIcmdWithAnInteger * pAllowedFieldCmd;
    //Source description file
    G4UIcmdWithAString * pSourceFileCmd;
    //Configuration of spot intensity as number of ions or MU (MU by default)
    G4UIcmdWithABool * pSpotIntensityCmd;
    //to inform the user about renamed option
    G4UIcmdWithABool * pDeprecatedSpotIntensityCmd;
    //Convergent or divergent beam model (divergent by default)
    G4UIcmdWithABool* pDivergenceCmd;
    G4UIcmdWithABool* pDivergenceXThetaCmd;
	 G4UIcmdWithABool* pDivergenceYPhiCmd;
    //Selection of one layer
    G4UIcmdWithAnInteger * pSelectLayerIDCmd;
    //Selection of one spot
    G4UIcmdWithAnInteger * pSelectSpotCmd;
};
// vim: ai sw=2 ts=2 et
#endif
