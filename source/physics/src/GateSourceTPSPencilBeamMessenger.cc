/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"
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
  //Particle Properties If GenericIon
  cmdName = GetDirectoryName()+"setIonProperties";
  pIonCmd = new G4UIcommand(cmdName,this);
  pIonCmd->SetGuidance("Set properties of ion to be generated:  Z:(int) AtomicNumber, A:(int) AtomicMass, Q:(int) Charge of Ion (in unit of e), E:(double) Excitation energy (in keV).");
  G4UIparameter* param;
  param = new G4UIparameter("Z",'i',false);
  param->SetDefaultValue("1");
  pIonCmd->SetParameter(param);
  param = new G4UIparameter("A",'i',false);
  param->SetDefaultValue("1");
  pIonCmd->SetParameter(param);
  param = new G4UIparameter("Q",'i',true);
  param->SetDefaultValue("0");
  pIonCmd->SetParameter(param);
  param = new G4UIparameter("E",'d',true);
  param->SetDefaultValue("0.0");
  pIonCmd->SetParameter(param);
  //Set the test Flag for debugging (verbosity)
  cmdName = GetDirectoryName()+"setTestFlag";
  pTestCmd = new G4UIcmdWithABool(cmdName,this);
  //Generate ions on random spots or by the same spot order as given in the plan
  cmdName = GetDirectoryName()+"setSortedSpotGenerationFlag";
  pSortedSpotGenerationCmd = new G4UIcmdWithABool(cmdName,this);
  //Choose absolute/relative energy spread (relative by default) specification (if not set in source properties file)
  cmdName = GetDirectoryName()+"setSigmaEnergyInMeVFlag";
  pSigmaEnergyInMeVCmd = new G4UIcmdWithABool(cmdName,this);
  //Treatment Plan file ("plan description file")
  cmdName = GetDirectoryName()+"setPlan";
  pPlanCmd = new G4UIcmdWithAString(cmdName,this);
  //FlatGenerationFlag
  cmdName = GetDirectoryName()+"setFlatGenerationFlag";
  pFlatGeneFlagCmd = new G4UIcmdWithABool(cmdName,this);
  //Not allowed fieldID
  cmdName = GetDirectoryName()+"setNotAllowedFieldID";
  pNotAllowedFieldCmd = new G4UIcmdWithAnInteger(cmdName,this);
  //Allowed fieldID
  cmdName = GetDirectoryName()+"setAllowedFieldID";
  pAllowedFieldCmd = new G4UIcmdWithAnInteger(cmdName,this);
  //Source description file
  cmdName = GetDirectoryName()+"setSourceDescriptionFile";
  pSourceFileCmd = new G4UIcmdWithAString(cmdName,this);
  //Configuration of spot intensity as number of ions or MU (MU by default)
  cmdName = GetDirectoryName()+"setSpotIntensityAsNbIons";
  pSpotIntensityCmd = new G4UIcmdWithABool(cmdName,this);
  //OLD configuration of spot intensity as number of ions or MU (to inform user about name change)
  cmdName = GetDirectoryName()+"setSpotIntensityAsNbProtons";
  pDeprecatedSpotIntensityCmd = new G4UIcmdWithABool(cmdName,this);
  //Convergent or divergent beam model (divergent by default)
  cmdName = GetDirectoryName()+"setBeamConvergence";
  pDivergenceCmd = new G4UIcmdWithABool(cmdName,this);
  cmdName = GetDirectoryName()+"setBeamConvergenceXTheta";
  pDivergenceXThetaCmd = new G4UIcmdWithABool(cmdName,this);
  cmdName = GetDirectoryName()+"setBeamConvergenceYPhi";
  pDivergenceYPhiCmd = new G4UIcmdWithABool(cmdName,this);
  //Selection of one layer
  cmdName = GetDirectoryName()+"selectLayerID";
  pSelectLayerIDCmd = new G4UIcmdWithAnInteger(cmdName,this);
  //Selection of one spot
  cmdName = GetDirectoryName()+"selectSpotID";
  pSelectSpotCmd = new G4UIcmdWithAnInteger(cmdName,this);
}

//----------------------------------------------------------------------------------------
GateSourceTPSPencilBeamMessenger::~GateSourceTPSPencilBeamMessenger()
{
  //FIXME seg fault?
  //delete pSourceTPSPencilBeam;

  //Particle Type
  delete pParticleTypeCmd;
  //Particle Properties If GenericIon
  delete pIonCmd;
  //Configuration of tests
  delete pTestCmd;
  //Sorted or random generation
  delete pSortedSpotGenerationCmd;
  //Absolute/relative energy spread specification
  delete pSigmaEnergyInMeVCmd;
  //Treatment Plan file
  delete pPlanCmd;
  //FlatGenerationFlag
  delete pFlatGeneFlagCmd;
  //Not allowed fieldID
  delete pNotAllowedFieldCmd;
  //Allowed fieldID
  delete pAllowedFieldCmd;
  //Source description file
  delete pSourceFileCmd;
  //Configuration of spot intensity
  delete pSpotIntensityCmd;
  //deprecated Configuration of spot intensity
  delete pDeprecatedSpotIntensityCmd;
  //Convergent or divergent beam model
  delete pDivergenceCmd;
  delete pDivergenceXThetaCmd;
  delete pDivergenceYPhiCmd;
  // Selection of one layer
  delete pSelectLayerIDCmd;
  // Selection of one spot
  delete pSelectSpotCmd;
}

//----------------------------------------------------------------------------------------
void GateSourceTPSPencilBeamMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  //Particle Type
  if (command == pParticleTypeCmd) {pSourceTPSPencilBeam->SetParticleType(newValue);}
  //Particle Properties If GenericIon
  if (command == pIonCmd) {
    pSourceTPSPencilBeam->SetIonParameter(newValue);
    pSourceTPSPencilBeam->SetIsGenericIon(true);
  }
  //Configuration of tests
  if (command == pTestCmd) {pSourceTPSPencilBeam->SetTestFlag(pTestCmd->GetNewBoolValue(newValue)); }
  //random or sorted spot generation
  if (command == pSortedSpotGenerationCmd) {pSourceTPSPencilBeam->SetSortedSpotGenerationFlag(pSortedSpotGenerationCmd->GetNewBoolValue(newValue)); }
  //Absolute/relative energy spread specification
  if (command == pSigmaEnergyInMeVCmd) {pSourceTPSPencilBeam->SetSigmaEnergyInMeVFlag(pSigmaEnergyInMeVCmd->GetNewBoolValue(newValue)); }
  //Treatment Plan file
  if (command == pPlanCmd) {pSourceTPSPencilBeam->SetPlan(newValue);  }
//  Configuration of FlatFlag gene
  if (command == pFlatGeneFlagCmd) {pSourceTPSPencilBeam->SetGeneFlatFlag(pFlatGeneFlagCmd->GetNewBoolValue(newValue)); }
  //Not allowed fieldID
  if (command == pNotAllowedFieldCmd) {pSourceTPSPencilBeam->SetNotAllowedField(pNotAllowedFieldCmd->GetNewIntValue(newValue));}
  //Allowed fieldID
  if (command == pAllowedFieldCmd) {pSourceTPSPencilBeam->SetAllowedField(pAllowedFieldCmd->GetNewIntValue(newValue));}
  //Select Layer ID
  if (command == pSelectLayerIDCmd) {pSourceTPSPencilBeam->SelectLayerID(pSelectLayerIDCmd->GetNewIntValue(newValue));}
  //Select Spot
  if (command == pSelectSpotCmd) {pSourceTPSPencilBeam->SelectSpot(pSelectSpotCmd->GetNewIntValue(newValue));}
  //Source description file
  if (command == pSourceFileCmd) {pSourceTPSPencilBeam->SetSourceDescriptionFile(newValue);  }
  //Configuration of spot intensity
  if (command == pSpotIntensityCmd) {pSourceTPSPencilBeam->SetSpotIntensity(pSpotIntensityCmd->GetNewBoolValue(newValue)); }
  if (command == pDeprecatedSpotIntensityCmd) {
    GateError("The 'setSpotIntensityAsNbProtons' option has been renamed 'setSpotIntensityAsNbIons'. Please update your macro file(s)!");
  }
  //Convergent or divergent beam model
  if (command == pDivergenceCmd) {pSourceTPSPencilBeam->SetBeamConvergence(pDivergenceCmd->GetNewBoolValue(newValue)); }
  if (command == pDivergenceXThetaCmd) {pSourceTPSPencilBeam->SetBeamConvergenceXTheta(pDivergenceCmd->GetNewBoolValue(newValue)); }
  if (command == pDivergenceYPhiCmd) {pSourceTPSPencilBeam->SetBeamConvergenceYPhi(pDivergenceCmd->GetNewBoolValue(newValue)); }
}
// vim: ai sw=2 ts=2 et
