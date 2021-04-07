/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateDetectorMessenger_h
#define GateDetectorMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

class GateDetectorConstruction;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithoutParameter;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithADouble;

class GateDetectorMessenger: public G4UImessenger
{
  public:
    GateDetectorMessenger(GateDetectorConstruction* );
    virtual ~GateDetectorMessenger();
    
    virtual void SetNewValue(G4UIcommand*, G4String);
    
  private:

    GateDetectorConstruction* pDetectorConstruction;
    
    G4UIdirectory*             pGateDir;
    G4UIdirectory*             pGateGeometryDir;
    
    G4UIcmdWithAString*        pMaterialDatabaseFilenameCmd;

    G4UIcmdWith3VectorAndUnit* pMagFieldCmd;
    G4UIcmdWithAString* 	   pMagTabulatedField3DCmd;

    G4UIcmdWith3VectorAndUnit* pElectFieldCmd;
    G4UIcmdWithAString* 	   pElectTabulatedField3DCmd;

    G4UIcmdWithAString* 	   pElectMagTabulatedField3DCmd;

    G4UIcmdWithAString*        pMagIntegratorStepperCmd;
    G4UIcmdWithADoubleAndUnit* pMagStepMinimumCmd;
    G4UIcmdWithADoubleAndUnit* pMagDeltaChordCmd;
    G4UIcmdWithADoubleAndUnit* pMagDeltaOneStepCmd;
    G4UIcmdWithADoubleAndUnit* pMagDeltaIntersectionCmd;
    G4UIcmdWithADouble*        pMagMinimumEpsilonStepCmd;
    G4UIcmdWithADouble*        pMagMaximumEpsilonStepCmd;

    G4UIcmdWithoutParameter*   pListCreatorsCmd;
    G4UIcmdWithAString*        IoniCmd;

    //G4UIcmdWithABool* 	       pEnableAutoUpdateCmd;    
    //G4UIcmdWithABool* 	       pDisableAutoUpdateCmd; 
    //G4UIcmdWithoutParameter*   pUpdateCmd;
    //G4UIcmdWithoutParameter*   pInitCmd;

};

#endif
