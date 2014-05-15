/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_LMF

#include "GateToLMFMessenger.hh"
#include "GateToLMF.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"



GateToLMFMessenger::GateToLMFMessenger(GateToLMF* gateToLMF)
  : GateOutputModuleMessenger(gateToLMF)
  , m_gateToLMF(gateToLMF)
{
  G4String cmdName;


  G4String cmdCoincidenceBool;
  G4String cmdDetectorIDBool;
  G4String cmdEnergyBool;
  G4String cmdNeighbourBool;
  G4String cmdNeighbourhoodOrder;
  G4String cmdGantryAxialPosBool;
  G4String cmdGantryAngularPosBool;
  G4String cmdSourcePosBool;
  G4String cmdGateDigiBool;
  G4String cmdComptonBool;
  //G4String cmdComptonDetectorBool;
  G4String cmdSourceIDBool;
  G4String cmdSourceXYZPosBool;
  G4String cmdGlobalXYZPosBool;
  G4String cmdEventIDBool;
  G4String cmdRunIDBool;








  cmdCoincidenceBool = GetDirectoryName()+"setCoincidenceBool";
  cmdDetectorIDBool = GetDirectoryName()+"setDetectorIDBool";
  cmdEnergyBool = GetDirectoryName()+"setEnergyBool";
  cmdNeighbourBool = GetDirectoryName()+"setNeighbourBool";
  cmdNeighbourhoodOrder = GetDirectoryName()+"setNeighbourhoodOrder";
  cmdGantryAxialPosBool = GetDirectoryName()+"setGantryAxialPosBool";
  cmdGantryAngularPosBool = GetDirectoryName()+"setGantryAngularPosBool";
  cmdSourcePosBool = GetDirectoryName()+"setSourcePosBool";
  cmdGateDigiBool = GetDirectoryName()+"setGateDigiBool";

  cmdComptonBool = GetDirectoryName()+"setComptonBool";
  //cmdComptonDetectorBool = GetDirectoryName()+"setComptonDetectorBool";
  cmdSourceIDBool = GetDirectoryName()+"setSourceIDBool";
  cmdSourceXYZPosBool = GetDirectoryName()+"setSourceXYZPosBool";
  cmdGlobalXYZPosBool = GetDirectoryName()+"setGlobalXYZPosBool";
  cmdEventIDBool = GetDirectoryName()+"setEventIDBool";
  cmdRunIDBool = GetDirectoryName()+"setRunIDBool";



  cmdName = GetDirectoryName()+"setFileName";
  GetFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  cmdName = GetDirectoryName()+"setInputDataName";
  SetInputDataCmd = new G4UIcmdWithAString(cmdName,this);

  GetCoincidenceBoolCmd = new G4UIcmdWithABool(cmdCoincidenceBool,this);
  GetDetectorIDBoolCmd = new G4UIcmdWithABool(cmdDetectorIDBool,this);
  GetEnergyBoolCmd = new G4UIcmdWithABool(cmdEnergyBool,this);
  GetNeighbourBoolCmd = new G4UIcmdWithABool(cmdNeighbourBool,this);
  GetNeighbourhoodOrderCmd = new G4UIcmdWithAnInteger(cmdNeighbourhoodOrder,this);
  GetGantryAxialPosBoolCmd = new G4UIcmdWithABool(cmdGantryAxialPosBool,this);
  GetGantryAngularPosBoolCmd = new G4UIcmdWithABool(cmdGantryAngularPosBool,this);
  GetSourcePosBoolCmd = new G4UIcmdWithABool(cmdSourcePosBool,this);
  GetGateDigiBoolCmd = new G4UIcmdWithABool(cmdGateDigiBool,this);
  GetComptonBoolCmd = new G4UIcmdWithABool(cmdComptonBool,this);
  //GetComptonDetectorBoolCmd = new G4UIcmdWithABool(cmdComptonDetectorBool,this);
  GetSourceIDBoolCmd = new G4UIcmdWithABool(cmdSourceIDBool,this);
  GetSourceXYZPosBoolCmd = new G4UIcmdWithABool(cmdSourceXYZPosBool,this);
  GetGlobalXYZPosBoolCmd = new G4UIcmdWithABool(cmdGlobalXYZPosBool,this);
  GetEventIDBoolCmd = new G4UIcmdWithABool(cmdEventIDBool,this);
  GetRunIDBoolCmd = new G4UIcmdWithABool(cmdRunIDBool,this);




  GetFileNameCmd->SetGuidance("Set the name of the file.ccs (LMF binary file)");
  GetFileNameCmd->SetGuidance("1.   Ex : MyCCSfile.ccs");
  SetInputDataCmd->SetGuidance("Set the name of the input data to store into the sinogram");



  GetCoincidenceBoolCmd->SetGuidance("Singles or coincidence store in LMF");
  GetDetectorIDBoolCmd->SetGuidance("Crystal ID stored or not in LMF");
  GetEnergyBoolCmd->SetGuidance("Energy stored or not in LMF");
  GetNeighbourBoolCmd->SetGuidance("Energy in neighbours stored or not in LMF");
  GetNeighbourhoodOrderCmd->SetGuidance("Neighbourhood order");
  GetGantryAxialPosBoolCmd->SetGuidance("GantryAxialPos stored or not in LMF");
  GetGantryAngularPosBoolCmd->SetGuidance("GantryAngularPosBool stored or not in LMF");
  GetSourcePosBoolCmd->SetGuidance("SourcePos stored or not in LMF");
  GetGateDigiBoolCmd->SetGuidance("Extended LMF gate digi record or not in LMF");
  GetComptonBoolCmd->SetGuidance("Number of Compton in phantom stored or not in LMF");
  //GetComptonDetectorBoolCmd->SetGuidance("Number of Compton in detector stored or not in LMF");
  GetSourceIDBoolCmd->SetGuidance("SourceID stored or not in LMF");
  GetSourceXYZPosBoolCmd->SetGuidance("SourceXYZPosBool stored or not in LMF");
  GetGlobalXYZPosBoolCmd->SetGuidance("GlobalXYZPos stored or not in LMF");
  GetEventIDBoolCmd->SetGuidance("EventID stored or not in LMF");
  GetRunIDBoolCmd->SetGuidance("RunID stored or not in LMF");

  SetInputDataCmd->SetParameterName("Name",false);

}



GateToLMFMessenger::~GateToLMFMessenger()
{
  delete GetFileNameCmd;
  delete SetInputDataCmd;




  delete GetCoincidenceBoolCmd;
  delete GetDetectorIDBoolCmd;
  delete GetEnergyBoolCmd ;
  delete GetNeighbourBoolCmd;
  delete GetNeighbourhoodOrderCmd;
  delete GetGantryAxialPosBoolCmd ;
  delete GetGantryAngularPosBoolCmd;
  delete GetSourcePosBoolCmd ;
  delete GetGateDigiBoolCmd;
  delete GetComptonBoolCmd ;
  //delete GetComptonDetectorBoolCmd ;
  delete GetSourceIDBoolCmd ;
  delete GetSourceXYZPosBoolCmd;
  delete GetGlobalXYZPosBoolCmd ;
  delete GetEventIDBoolCmd ;
  delete GetRunIDBoolCmd ;

}



void GateToLMFMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command == GetFileNameCmd )
    m_gateToLMF->SetOutputFileName(newValue);
  else if (command == SetInputDataCmd)
    { m_gateToLMF->SetOutputDataName(newValue); }
  else if (command == GetCoincidenceBoolCmd)
    m_gateToLMF->SetCoincidenceBool(GetCoincidenceBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetDetectorIDBoolCmd)
    m_gateToLMF->SetDetectorIDBool(GetDetectorIDBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetEnergyBoolCmd)
    m_gateToLMF->SetEnergyBool(GetEnergyBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetNeighbourBoolCmd)
    m_gateToLMF->SetNeighbourBool(GetNeighbourBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetNeighbourhoodOrderCmd)
    m_gateToLMF->SetNeighbourhoodOrder(GetNeighbourhoodOrderCmd->GetNewIntValue(newValue));
  else if (command == GetGantryAxialPosBoolCmd )
    m_gateToLMF->SetGantryAxialPosBool(GetGantryAxialPosBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetGantryAngularPosBoolCmd )
    m_gateToLMF->SetGantryAngularPosBool(GetGantryAngularPosBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetSourcePosBoolCmd )
    m_gateToLMF->SetSourcePosBool(GetSourcePosBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetGateDigiBoolCmd)
    m_gateToLMF->SetGateDigiBool(GetGateDigiBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetComptonBoolCmd)
    m_gateToLMF->SetComptonBool(GetComptonBoolCmd->GetNewBoolValue(newValue));
  //else if (command == GetComptonDetectorBoolCmd)
    //m_gateToLMF->SetComptonDetectorBool(GetComptonDetectorBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetSourceIDBoolCmd)
    m_gateToLMF->SetSourceIDBool(GetSourceIDBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetSourceXYZPosBoolCmd)
    m_gateToLMF->SetSourceXYZPosBool(GetSourceXYZPosBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetGlobalXYZPosBoolCmd)
    m_gateToLMF->SetGlobalXYZPosBool(GetGlobalXYZPosBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetEventIDBoolCmd)
    m_gateToLMF->SetEventIDBool(GetEventIDBoolCmd->GetNewBoolValue(newValue));
  else if (command == GetRunIDBoolCmd)
    m_gateToLMF->SetRunIDBool(GetRunIDBoolCmd->GetNewBoolValue(newValue));
  else
    GateOutputModuleMessenger::SetNewValue(command,newValue);


}


#endif
