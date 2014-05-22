/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
  \class  GateHounsfieldToMaterialsBuilderMessenger.cc
  \brief  
  \author david.sarrut@creatis.insa-lyon.fr
*/
 
#include "GateHounsfieldToMaterialsBuilderMessenger.hh"
#include "GateHounsfieldToMaterialsBuilder.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

//-----------------------------------------------------------------------------------------
GateHounsfieldToMaterialsBuilderMessenger::GateHounsfieldToMaterialsBuilderMessenger(GateHounsfieldToMaterialsBuilder * m)
{
  mBuilder = m; 

  // Create folder
  G4String dir = "/gate/HounsfieldMaterialGenerator/";
  G4String cmdName;
  
  // Create commands
  cmdName = dir+"Generate";
  pGenerateCmd = new G4UIcmdWithoutParameter(cmdName.c_str(),this);
  pGenerateCmd->SetGuidance("Generate the two output files (material DB & correspondance between HU and material");

  cmdName = dir+"SetMaterialTable";
  pSetMaterialTable = new G4UIcmdWithAString(cmdName.c_str(),this);

  cmdName = dir+"SetDensityTable";
  pSetDensityTable = new G4UIcmdWithAString(cmdName.c_str(),this);

  cmdName = dir+"SetOutputMaterialDatabaseFilename";
  pSetOutputMaterialDatabaseFilename = new G4UIcmdWithAString(cmdName.c_str(),this);

  cmdName = dir+"SetOutputHUMaterialFilename";
  pSetOutputHUMaterialFilename = new G4UIcmdWithAString(cmdName.c_str(),this);

  cmdName = dir+"SetDensityTolerance";
  pSetDensityTolerance = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);

}
//-----------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------
GateHounsfieldToMaterialsBuilderMessenger::~GateHounsfieldToMaterialsBuilderMessenger()
{
  delete pGenerateCmd;
  delete pSetMaterialTable;
  delete pSetDensityTable;
  delete pSetOutputMaterialDatabaseFilename;
  delete pSetOutputHUMaterialFilename;
  delete pSetDensityTolerance;
}
//-----------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------
void GateHounsfieldToMaterialsBuilderMessenger::SetNewValue(G4UIcommand *c, G4String s) {  
  if (c == pSetMaterialTable) mBuilder->SetMaterialTable(s);
  if (c == pSetDensityTable)  mBuilder->SetDensityTable(s);
  if (c == pSetOutputMaterialDatabaseFilename)  mBuilder->SetOutputMaterialDatabaseFilename(s);
  if (c == pSetOutputHUMaterialFilename)  mBuilder->SetOutputHUMaterialFilename(s);
  if (c == pSetDensityTolerance)  mBuilder->SetDensityTolerance(pSetDensityTolerance->GetNewDoubleValue(s));
  if (c == pGenerateCmd) mBuilder->BuildAndWriteMaterials();  
}
//-----------------------------------------------------------------------------------------


