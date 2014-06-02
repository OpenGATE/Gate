/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT
#ifndef GateSourcePencilBeamMessenger_h
#define GateSourcePencilBeamMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "GateMessenger.hh"
#include "GateVSourceMessenger.hh"

class GateSourcePencilBeam;
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

class GateSourcePencilBeamMessenger: public GateVSourceMessenger
{
  public:
    GateSourcePencilBeamMessenger(GateSourcePencilBeam* source);
    ~GateSourcePencilBeamMessenger();
    void SetNewValue(G4UIcommand*, G4String);
  private:
    GateSourcePencilBeam * pSourcePencilBeam;
    //Particle Type
    G4UIcmdWithAString * pParticleTypeCmd;
    //Particle Properties If GenericIon
    G4UIcommand * pIonCmd;
    //Energy
    G4UIcmdWithADoubleAndUnit * pEnergyCmd;
    G4UIcmdWithADoubleAndUnit * pSigmaEnergyCmd;
    //Position
    G4UIcmdWith3VectorAndUnit * pPositionCmd;
    G4UIcmdWithADoubleAndUnit * pSigmaXCmd;
    G4UIcmdWithADoubleAndUnit * pSigmaYCmd;
    //Direction
    G4UIcmdWithADoubleAndUnit * pSigmaThetaCmd;
    G4UIcmdWithADoubleAndUnit * pSigmaPhiCmd;
    G4UIcmdWith3Vector * pRotationAxisCmd;
    G4UIcmdWithADoubleAndUnit * pRotationAngleCmd;
    //Correlation Position/Direction
    G4UIcmdWithADoubleAndUnit * pEllipseXThetaAreaCmd;
    G4UIcmdWithADoubleAndUnit * pEllipseYPhiAreaCmd;
    G4UIcmdWithAString * pEllipseXThetaRotationNormCmd;
    G4UIcmdWithAString * pEllipseYPhiRotationNormCmd;
    //Tests
    G4UIcmdWithABool* pTestCmd;
};

#endif

#endif

