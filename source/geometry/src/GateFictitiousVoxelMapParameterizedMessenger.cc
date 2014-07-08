/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateFictitiousVoxelMapParameterizedMessenger.hh"
#include "GateFictitiousVoxelMapParameterized.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "GatePETVRTManager.hh"
#include "GatePETVRTSettings.hh"

///////////////////
//  Constructor  //
///////////////////

GateFictitiousVoxelMapParameterizedMessenger::GateFictitiousVoxelMapParameterizedMessenger(GateFictitiousVoxelMapParameterized *itsInserter)
  :GateMessenger(itsInserter->GetObjectName()+"/geometry"),m_inserter(itsInserter)
{
  GetDirectory()->SetGuidance("Control the fictitious voxel map geometry.");
  G4String cmdName;

  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+ "/attachVoxelPhantomSD";
  AttachPhantomSDCmd = new G4UIcmdWithoutParameter(cmdName,this);
  AttachPhantomSDCmd->SetGuidance("Attach the phantom-SD to the matrix.");

  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+ "/addOutput";
  AddOutputCmd = new G4UIcmdWithAString(cmdName,this);
  AddOutputCmd->SetGuidance("Adds an output module to write the dose matrix");

  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+ "/verbose";
  VerboseCmd = new G4UIcmdWithAnInteger(cmdName,this);
  VerboseCmd->SetGuidance("Set the verbosity level of the construction of the fictitious voxel map phantom, from 1 to 5 (default is 0)");

  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+"/setSkipEqualMaterials";
  SkipEqualMaterialsCmd = new G4UIcmdWithABool(cmdName,this);
  SkipEqualMaterialsCmd->SetGuidance("Skip or not boundaries when neighbour voxels are made of same material (default: yes)");

  cmdName = GetDirectoryName()+"insertReader";
  InsertReaderCmd = new G4UIcmdWithAString(cmdName,this);
  InsertReaderCmd->SetGuidance("Insert a reader of the type specified");

  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+"/setFictitiousEnergy";
  FictitiousEnergyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  FictitiousEnergyCmd->SetGuidance("Set gamma energy above that fictitious interaction tracking is used for the voxelized phantom");
  FictitiousEnergyCmd->SetParameterName("fenergy",false);
  FictitiousEnergyCmd->SetRange("fenergy>0.");
  FictitiousEnergyCmd->SetUnitCategory("Energy");
//  FictitiousEnergyCmd->AvailableForStates(G4State_PreInit);

  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+"/setGammaDiscardEnergy";
  DiscardEnergyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  DiscardEnergyCmd->SetGuidance("Set gamma energy threshold. If gamma energy is below this threshold after interaction in the phantom, the particle is discarded.");
  DiscardEnergyCmd->SetParameterName("denergy",false);
  DiscardEnergyCmd->SetRange("denergy>0.");
  DiscardEnergyCmd->SetUnitCategory("Energy");
//  DiscardEnergyCmd->AvailableForStates(G4State_PreInit);

  cmdName = GetDirectoryName()+"removeReader";
  RemoveReaderCmd = new G4UIcmdWithoutParameter(cmdName,this);
  RemoveReaderCmd->SetGuidance("Remove the reader");
}

//////////////////
//  Destructor  //
//////////////////

GateFictitiousVoxelMapParameterizedMessenger::~GateFictitiousVoxelMapParameterizedMessenger()
{
   delete InsertReaderCmd;
   delete RemoveReaderCmd;
   delete AttachPhantomSDCmd;
   delete AddOutputCmd;
   delete VerboseCmd;
   delete DiscardEnergyCmd;
   delete FictitiousEnergyCmd;
   delete SkipEqualMaterialsCmd;
}

///////////////////
//  SetNewValue  //
///////////////////

void GateFictitiousVoxelMapParameterizedMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command==AttachPhantomSDCmd )
    { GetFictitiousVoxelMapParameterized()->AttachPhantomSD(); }

  else if ( command == AddOutputCmd )
    { GetFictitiousVoxelMapParameterized()->AddOutput(newValue); }

  else if ( command == VerboseCmd )
    { GetFictitiousVoxelMapParameterized()->SetVerbosity(VerboseCmd->GetNewIntValue(newValue)); }

  else if ( command == SkipEqualMaterialsCmd )
    { GetFictitiousVoxelMapParameterized()->ChangeSkipEqualMaterials(SkipEqualMaterialsCmd->GetNewBoolValue(newValue)); }

  else if ( command == InsertReaderCmd )
    { GetFictitiousVoxelMapParameterized()->InsertReader(newValue); }

  else if ( command == RemoveReaderCmd )
    { GetFictitiousVoxelMapParameterized()->RemoveReader(); }

  else if (command == FictitiousEnergyCmd)
    { GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->SetFictitiousEnergy(FictitiousEnergyCmd->GetNewDoubleValue(newValue)); }

  else if (command == DiscardEnergyCmd)
    { GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->SetDiscardEnergy(DiscardEnergyCmd->GetNewDoubleValue(newValue)); }

  else
    GateMessenger::SetNewValue(command,newValue);
}
