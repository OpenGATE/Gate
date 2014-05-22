/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
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
 
  G4double turn   = 2. * M_PI * radian;

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




