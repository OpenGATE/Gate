/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateConfiguration.h"

#include "GateDetectorMessenger.hh"
#include "GateDetectorConstruction.hh"

#include "GateDistributionListManager.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UnitsTable.hh"
#include "G4Material.hh"
#include "GateMiscFunctions.hh"

#include "GateObjectStore.hh"

//-----------------------------------------------------------------------------
GateDetectorMessenger::GateDetectorMessenger(GateDetectorConstruction* GateDet)
:pDetectorConstruction(GateDet)
{ 

  G4double minute = 60.* second;
  G4double hour   = 60.* minute;

  //We build a table of speed units
  new G4UnitDefinition(     "m/s","m/s"       ,"Speed",meter/second); 
  new G4UnitDefinition(    "cm/s","cm/s"      ,"Speed",centimeter/second); 
  new G4UnitDefinition(    "mm/s","mm/s"      ,"Speed",millimeter/second); 
  new G4UnitDefinition(   "m/min","m/min"     ,"Speed",meter/minute); 
  new G4UnitDefinition(  "cm/min","cm/min"    ,"Speed",centimeter/minute); 
  new G4UnitDefinition(  "mm/min","mm/min"    ,"Speed",millimeter/minute); 
  new G4UnitDefinition(     "m/h","m/h"       ,"Speed",meter/hour); 
  new G4UnitDefinition(    "cm/h","cm/h"      ,"Speed",centimeter/hour); 
  new G4UnitDefinition(    "mm/h","mm/h"      ,"Speed",millimeter/hour); 
 
  //We build a table of LET units
  
  new G4UnitDefinition( "keV/um", "keV/um", "Energy/Length", keV/um );
  new G4UnitDefinition( "MeV/mm", "MeV/mm", "Energy/Length", MeV/mm );
  new G4UnitDefinition( "keV/mm", "keV/mm", "Energy/Length", keV/mm );
  new G4UnitDefinition( "MeV/um", "MeV/um", "Energy/Length", MeV/um );
  
  G4double turn   = twopi * radian;

  //We build a table of angular speed units
  new G4UnitDefinition(    "radian/s","rad/s"      ,"Angular speed",radian/second); 
  new G4UnitDefinition(    "degree/s","deg/s"      ,"Angular speed",degree/second); 
  new G4UnitDefinition(  "rotation/s","rot/s"      ,"Angular speed",turn/second); 
  new G4UnitDefinition(  "radian/min","rad/min"    ,"Angular speed",radian/minute); 
  new G4UnitDefinition(  "degree/min","deg/min"    ,"Angular speed",degree/minute); 
  new G4UnitDefinition("rotation/min","rot/min"    ,"Angular speed",turn/minute); 
  new G4UnitDefinition(    "radian/h","rad/h"      ,"Angular speed",radian/hour); 
  new G4UnitDefinition(    "degree/h","deg/h"      ,"Angular speed",degree/hour); 
  new G4UnitDefinition(  "rotation/h","rot/h"      ,"Angular speed",turn/hour); 
 
  //We build a table of computer memory size
  new G4UnitDefinition(    "byte","B"      ,"Memory size",1L);
  new G4UnitDefinition(    "kilobyte","kB"      ,"Memory size",1L<<10);
  new G4UnitDefinition(    "megabyte","MB"      ,"Memory size",1L<<20);
  new G4UnitDefinition(    "gigabyte","GB"      ,"Memory size",1L<<30);
  
  // Add units for light yield
#ifdef GATE_USE_OPTICAL
  new G4UnitDefinition("1/eV",  "1/eV",  "per energy", 1.0/eV);
  new G4UnitDefinition("1/keV", "1/keV", "per energy", 1.0/keV);
  new G4UnitDefinition("1/MeV", "1/MeV", "per energy", 1.0/MeV);
  new G4UnitDefinition("1/GeV", "1/GeV", "per energy", 1.0/GeV);
#endif

  pGateDir = new G4UIdirectory("/gate/");
//  pGateDir->SetGuidance("UI commands of this example");
  pGateDir->SetGuidance("GATE detector control.");
  
  pGateGeometryDir = new G4UIdirectory("/gate/geometry/");
  pGateGeometryDir->SetGuidance("Gate geometry control.");

  pMagFieldCmd = new G4UIcmdWith3VectorAndUnit("/gate/geometry/setMagField",this);  
  pMagFieldCmd->SetGuidance("Define magnetic field.");
  pMagFieldCmd->SetParameterName("Bx","By","Bz",false);
  pMagFieldCmd->SetUnitCategory("Magnetic flux density");
  pMagFieldCmd->SetDefaultUnit("tesla");

  pMagTabulatedField3DCmd = new G4UIcmdWithAString("/gate/geometry/setMagTabulateField3D",this);
  pMagTabulatedField3DCmd->SetGuidance("Sets the data filename of magnetic field 3D.");
  pMagTabulatedField3DCmd->SetParameterName(" Magnetic field tabulate filename ",false);

  pElectFieldCmd = new G4UIcmdWith3VectorAndUnit("/gate/geometry/setElectField",this);
  pElectFieldCmd->SetGuidance("Define electric field.");
//  pElectFieldCmd->SetParameterName("Ex","Ey","Ez",false);
//  pElectFieldCmd->SetUnitCategory("Electric flux density");
//  pElectFieldCmd->SetDefaultUnit("volt");

  pElectTabulatedField3DCmd = new G4UIcmdWithAString("/gate/geometry/setElectTabulateField3D",this);
  pElectTabulatedField3DCmd->SetGuidance("Sets the data filename of electric field 3D.");
  pElectTabulatedField3DCmd->SetParameterName(" Electric field tabulate filename ",false);

  pElectMagTabulatedField3DCmd = new G4UIcmdWithAString("/gate/geometry/setElectMagTabulateField3D",this);
  pElectMagTabulatedField3DCmd->SetGuidance("Sets the data filename of electromagnetic field 3D.");
  pElectMagTabulatedField3DCmd->SetParameterName(" Electromagnetic field tabulate filename ",false);

  G4String dir = "/gate/geometry/setMagTabulateField3D/";
  G4String cmdName;

  cmdName = dir+"setStepMinimum";
  pMagStepMinimumCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pMagStepMinimumCmd->SetGuidance("Set a minimum step size used during integration"
          "to compute the motion of a charged track in a general field.");
  pMagStepMinimumCmd->SetUnitCategory("Length");
  pMagStepMinimumCmd->SetDefaultUnit("m");

  cmdName = dir+"setMissDistance";
  pMagDeltaChordCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pMagDeltaChordCmd->SetGuidance("Set a miss distance between the 'real' "
		  "curved trajectory and the approximate linear trajectory of the chord");
  pMagDeltaChordCmd->SetUnitCategory("Length");
  pMagDeltaChordCmd->SetDefaultUnit("m");

  cmdName = dir+"setDeltaIntersection";
  pMagDeltaIntersectionCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pMagDeltaIntersectionCmd->SetGuidance(" Set the accuracy to which an intersection"
		  " with a volume boundary is calculated.");
  pMagDeltaIntersectionCmd->SetUnitCategory("Length");
  pMagDeltaIntersectionCmd->SetDefaultUnit("m");

  cmdName = dir+"setDeltaOneStep";
  pMagDeltaOneStepCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pMagDeltaOneStepCmd->SetGuidance("Set a limit on the estimated error of the endpoint"
		  " of each physics step.");
  pMagDeltaOneStepCmd->SetUnitCategory("Length");
  pMagDeltaOneStepCmd->SetDefaultUnit("m");

  cmdName = dir+"setMinimumEpsilonStep";
  pMagMinimumEpsilonStepCmd = new G4UIcmdWithADouble(cmdName.c_str(),this);
  pMagMinimumEpsilonStepCmd->SetGuidance("Impose a minimum limit on the relative error"
		  " of the position/momentum inaccuracy.");

  cmdName = dir+"setMaximumEpsilonStep";
  pMagMaximumEpsilonStepCmd = new G4UIcmdWithADouble(cmdName.c_str(),this);
  pMagMaximumEpsilonStepCmd->SetGuidance("Impose a maximum limit on the relative error"
		  " of the position/momentum inaccuracy.");

  cmdName = dir+"setIntegratorStepper";
  pMagIntegratorStepperCmd = new G4UIcmdWithAString(cmdName,this);
  pMagIntegratorStepperCmd->SetGuidance("Set integrator stepper to compute the motion"
		  " of a charged track in a general field.");
  pMagIntegratorStepperCmd->SetParameterName(" Integrator Stepper Type ",false);

  G4String cmd;
  cmd = "/gate/geometry/setMaterialDatabase";
  pMaterialDatabaseFilenameCmd = new G4UIcmdWithAString(cmd, this);
  pMaterialDatabaseFilenameCmd->SetGuidance("Sets the filename of the material database to use");
  pMaterialDatabaseFilenameCmd->SetParameterName("Material database filename", true);
  
  pListCreatorsCmd = new G4UIcmdWithoutParameter("/gate/geometry/listVolumes",this);
  pListCreatorsCmd->SetGuidance("List all the volume creators in the GATE geometry");

  cmd = "/gate/geometry/setIonisationPotential";
  IoniCmd = new G4UIcmdWithAString(cmd,this);
  IoniCmd->SetGuidance("Set the ionisation potential for a material (two parameters 'material' and 'value and unit')");




  GateDistributionListManager::Init();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateDetectorMessenger::~GateDetectorMessenger()
{
  delete pMaterialDatabaseFilenameCmd;
  delete pMagFieldCmd;
  delete pListCreatorsCmd;
  delete IoniCmd;

  delete pGateGeometryDir;
  delete pGateDir;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{   
   
  if (command == pMaterialDatabaseFilenameCmd ) {
      pDetectorConstruction->AddFileToMaterialDatabase(newValue);
  }
  else if( command == pMagFieldCmd )
    { pDetectorConstruction->SetMagField(pMagFieldCmd->GetNew3VectorValue(newValue));}

  else if( command == pMagTabulatedField3DCmd )
    { pDetectorConstruction->SetMagFieldTabulatedFile(newValue);}

  else if( command == pElectFieldCmd )
    { pDetectorConstruction->SetElectField(pMagFieldCmd->GetNew3VectorValue(newValue));}

  else if( command == pElectTabulatedField3DCmd )
    { pDetectorConstruction->SetElectFieldTabulatedFile(newValue);}

  else if( command == pElectMagTabulatedField3DCmd )
    { pDetectorConstruction->SetElectMagFieldTabulatedFile(newValue);}

  else if( command == pMagStepMinimumCmd )
    { pDetectorConstruction->SetMagStepMinimum(pMagStepMinimumCmd->GetNewDoubleValue(newValue));}

  else if( command == pMagDeltaChordCmd )
    { pDetectorConstruction->SetMagDeltaChord(pMagDeltaChordCmd->GetNewDoubleValue(newValue));}

  else if( command == pMagDeltaOneStepCmd )
    { pDetectorConstruction->SetMagDeltaOneStep(pMagDeltaOneStepCmd->GetNewDoubleValue(newValue));}

  else if( command == pMagDeltaIntersectionCmd )
    { pDetectorConstruction->SetMagDeltaIntersection(pMagDeltaIntersectionCmd->GetNewDoubleValue(newValue));}

  else if( command == pMagMinimumEpsilonStepCmd )
    { pDetectorConstruction->SetMagMinimumEpsilonStep(pMagMinimumEpsilonStepCmd->GetNewDoubleValue(newValue));}

  else if( command == pMagMaximumEpsilonStepCmd )
    { pDetectorConstruction->SetMagMaximumEpsilonStep(pMagMaximumEpsilonStepCmd->GetNewDoubleValue(newValue));}

  else if( command == pMagIntegratorStepperCmd )
     { pDetectorConstruction->SetMagIntegratorStepper(newValue);}
 
  else if( command == pListCreatorsCmd )
    { pDetectorConstruction->GetObjectStore()->ListCreators(); }
  else if( command == IoniCmd )
    {
      G4String matName;
      double value;
      GetStringAndValueFromCommand(command, newValue, matName, value);
      pDetectorConstruction->SetMaterialIoniPotential(matName,value);
    }
  else
    G4UImessenger::SetNewValue(command,newValue);
    
}
//-----------------------------------------------------------------------------




