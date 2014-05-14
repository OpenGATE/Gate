/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#ifndef GATESOURCETPSPENCILBEAMMESSENGER_CC
#define GATESOURCETPSPENCILBEAMMESSENGER_CC

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT
#include "GateSourceTPSPencilBeamMessenger.hh"
#include "GateSourceTPSPencilBeam.hh"
#include "GateUIcmdWithTwoDouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UnitsTable.hh"



//----------------------------------------------------------------------------------------
  GateSourceTPSPencilBeamMessenger::GateSourceTPSPencilBeamMessenger(GateSourceTPSPencilBeam* source)
: GateVSourceMessenger(source)
{
  pSourceTPSPencilBeam = source;
  G4String cmdName;

  //Particle Type
  cmdName = GetDirectoryName()+"setParticleType";
  pParticleTypeCmd = new G4UIcmdWithAString(cmdName,this);
  //Configuration of tests
  cmdName = GetDirectoryName()+"setTestFlag";
  pTestCmd = new G4UIcmdWithABool(cmdName,this);
  //Treatment Plan file
  cmdName = GetDirectoryName()+"setPlan";
  pPlanCmd = new G4UIcmdWithAString(cmdName,this);
  //FlatGenerationFlag
  cmdName = GetDirectoryName()+"setFlatGenerationFlag";
  pFlatGeneFlagCmd = new G4UIcmdWithABool(cmdName,this);
  //Not allowed fieldID
  cmdName = GetDirectoryName()+"setNotAllowedFieldID";
  pNotAllowedFieldCmd = new G4UIcmdWithAnInteger(cmdName,this);
  //Source description file
  cmdName = GetDirectoryName()+"setSourceDescriptionFile";
  pSourceFileCmd = new G4UIcmdWithAString(cmdName,this);
  //Configuration of spot intensity
  cmdName = GetDirectoryName()+"setSpotIntensityAsNbProtons";
  pSpotIntensityCmd = new G4UIcmdWithABool(cmdName,this);
  //Convergent or divergent beam model
  cmdName = GetDirectoryName()+"setBeamConvergence";
  pDivergenceCmd = new G4UIcmdWithABool(cmdName,this);
  //Selection of one layer
  cmdName = GetDirectoryName()+"selectLayerID";
  pSelectLayerIDCmd = new G4UIcmdWithAnInteger(cmdName,this);
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateSourceTPSPencilBeamMessenger::~GateSourceTPSPencilBeamMessenger()
{
  //delete pSourceTPSPencilBeam;
  //Particle Type
  delete pParticleTypeCmd;
  //Configuration of tests
  delete pTestCmd;
  //Treatment Plan file
  delete pPlanCmd;
  //FlatGenerationFlag
  delete pFlatGeneFlagCmd;
  //Not allowed fieldID
  delete pNotAllowedFieldCmd;
  //Source description file
  delete pSourceFileCmd;
  //Configuration of spot intensity
  delete pSpotIntensityCmd;
  //Convergent or divergent beam model
  delete pDivergenceCmd;
  // Selection of one layer
  delete pSelectLayerIDCmd;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourceTPSPencilBeamMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  //Particle Type
  if (command == pParticleTypeCmd) {pSourceTPSPencilBeam->SetParticleType(newValue);  }
  //Configuration of tests
  if (command == pTestCmd) {pSourceTPSPencilBeam->SetTestFlag(pTestCmd->GetNewBoolValue(newValue)); }
  //Treatment Plan file
  if (command == pPlanCmd) {pSourceTPSPencilBeam->SetPlan(newValue);  }
  //Configuration of FlatFlag gene
  if (command == pFlatGeneFlagCmd) {pSourceTPSPencilBeam->SetGeneFlatFlag(pFlatGeneFlagCmd->GetNewBoolValue(newValue)); }
  //Not allowed fieldID
  if (command == pNotAllowedFieldCmd) {pSourceTPSPencilBeam->SetNotAllowedField(pNotAllowedFieldCmd->GetNewIntValue(newValue));}
  //Select Layer ID
  if (command == pSelectLayerIDCmd) {pSourceTPSPencilBeam->SelectLayerID(pSelectLayerIDCmd->GetNewIntValue(newValue));}
  //Source description file
  if (command == pSourceFileCmd) {pSourceTPSPencilBeam->SetSourceDescriptionFile(newValue);  }
  //Configuration of spot intensity
  if (command == pSpotIntensityCmd) {pSourceTPSPencilBeam->SetSpotIntensity(pSpotIntensityCmd->GetNewBoolValue(newValue)); }
  //Convergent or divergent beam model
  if (command == pDivergenceCmd) {pSourceTPSPencilBeam->SetBeamConvergence(pDivergenceCmd->GetNewBoolValue(newValue)); }
}
#endif

#endif
