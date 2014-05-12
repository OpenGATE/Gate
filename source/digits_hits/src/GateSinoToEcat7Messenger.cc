/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*----------------------
   Modifications history

     Gate 6.2

	C. Comtat, CEA/SHFJ, 10/02/2011	   Allows for span 1 (means less slices per segment)

                                           Allows for an interfile-like ("ecat8") output instead of ecat7.
					   It does not require the ecat library! (GATE_USE_ECAT7 not set)
----------------------*/

#include "GateConfiguration.h"

#include "GateSinoToEcat7Messenger.hh"
#include "GateSinoToEcat7.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"





GateSinoToEcat7Messenger::GateSinoToEcat7Messenger(GateSinoToEcat7* gateSinoToEcat7)
  : GateOutputModuleMessenger(gateSinoToEcat7)
  , m_gateSinoToEcat7(gateSinoToEcat7)
{
  G4String cmdName;

  cmdName = GetDirectoryName()+"setFileName";
  SetFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetFileNameCmd->SetGuidance("Set the name of the output ECAT7 sinogram file");
  SetFileNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"mashing";
  SetMashingCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetMashingCmd->SetGuidance("Set azimutal mashing factor.");
  SetMashingCmd->SetParameterName("Number",false);
  SetMashingCmd->SetRange("Number>0");

  cmdName = GetDirectoryName()+"span";
  SetSpanCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetSpanCmd->SetGuidance("Set span (polar mashing) factor.");
  SetSpanCmd->SetParameterName("Number",false);
  SetSpanCmd->SetRange("Number>0");

  cmdName = GetDirectoryName()+"maxringdiff";
  SetMaxRingDiffCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetMaxRingDiffCmd->SetGuidance("Set maximum ring difference.");
  SetMaxRingDiffCmd->SetParameterName("Number",false);
  SetMaxRingDiffCmd->SetRange("Number>0");

  cmdName = GetDirectoryName()+"system";
  SetEcatCameraNumberCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetEcatCameraNumberCmd->SetGuidance("Set camera type according to ECAT numerotation.");
  SetEcatCameraNumberCmd->SetParameterName("Number",false);
  SetEcatCameraNumberCmd->SetRange("Number>0");

  cmdName = GetDirectoryName()+"IsotopeCode";
  SetIsotopeCodeCmd = new G4UIcmdWithAString(cmdName,this);
  SetIsotopeCodeCmd->SetGuidance("Set isotope-code for the ecat7 main header only");
  SetFileNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"IsotopeHalflife";
  SetIsotopeHalflifeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  SetIsotopeHalflifeCmd->SetGuidance("Set isotope half-life for the ecat7 main header only");
  SetIsotopeHalflifeCmd->SetParameterName("Number",false);
  SetIsotopeHalflifeCmd->SetUnitCategory("Time");

  cmdName = GetDirectoryName()+"IsotopeBranchingFraction";
  SetIsotopeBranchingFractionCmd = new G4UIcmdWithADouble(cmdName,this);
  SetIsotopeBranchingFractionCmd->SetGuidance("Set isotope branching fraction for the ecat7 main header only");
  SetIsotopeBranchingFractionCmd->SetParameterName("Number",false);
  SetIsotopeBranchingFractionCmd->SetRange("Number<=1.0");

  #ifdef GATE_USE_ECAT7
  cmdName = GetDirectoryName()+"version";
  SetEcatVersionCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetEcatVersionCmd->SetGuidance("Set ecat version (7 for ecat7 or 8 for interfile-like");
  SetEcatVersionCmd->SetParameterName("Number",false);
  SetEcatVersionCmd->SetRange("Number>6 && Number<9");
  #endif
}





GateSinoToEcat7Messenger::~GateSinoToEcat7Messenger()
{
  delete SetFileNameCmd;
  delete SetMashingCmd;
  delete SetSpanCmd;
  delete SetMaxRingDiffCmd;
  delete SetEcatCameraNumberCmd;
  delete SetIsotopeCodeCmd;
  delete SetIsotopeHalflifeCmd;
  delete SetIsotopeBranchingFractionCmd;
  #ifdef GATE_USE_ECAT7
  delete SetEcatVersionCmd;
  #endif
}





void GateSinoToEcat7Messenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if (command == SetFileNameCmd)
    { m_gateSinoToEcat7->SetFileName(newValue); }
  else if ( command==SetMashingCmd )
    { m_gateSinoToEcat7->SetMashing(SetMashingCmd->GetNewIntValue(newValue)); }
  else if ( command==SetSpanCmd )
    { m_gateSinoToEcat7->SetSpan(SetSpanCmd->GetNewIntValue(newValue)); }
  else if ( command==SetMaxRingDiffCmd )
    { m_gateSinoToEcat7->SetMaxRingDiff(SetMaxRingDiffCmd->GetNewIntValue(newValue)); }
  else if ( command==SetEcatCameraNumberCmd )
    { m_gateSinoToEcat7->SetEcatCameraNumber(SetEcatCameraNumberCmd->GetNewIntValue(newValue)); }
  else if ( command==SetIsotopeCodeCmd )
    { m_gateSinoToEcat7->SetIsotopeCode(newValue); }
  else if ( command==SetIsotopeHalflifeCmd )
    { m_gateSinoToEcat7->SetIsotopeHalflife(SetIsotopeHalflifeCmd->GetNewDoubleValue(newValue)); }
  else if ( command==SetIsotopeBranchingFractionCmd )
    { m_gateSinoToEcat7->SetIsotopeBranchingFraction(SetIsotopeBranchingFractionCmd->GetNewDoubleValue(newValue)); }
  #ifdef GATE_USE_ECAT7
  else if ( command==SetEcatVersionCmd )
    { m_gateSinoToEcat7->SetEcatVersion(SetEcatVersionCmd->GetNewIntValue(newValue)); }
  #endif
  else
    { GateOutputModuleMessenger::SetNewValue(command,newValue);  }

}
