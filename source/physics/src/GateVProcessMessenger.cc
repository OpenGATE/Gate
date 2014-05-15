/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEPROCESSBUILDERMESSENGER_CC
#define GATEPROCESSBUILDERMESSENGER_CC


#include "GateVProcess.hh"
#include "GateVProcessMessenger.hh"

#include "G4UIdirectory.hh"

//-----------------------------------------------------------------------------
GateVProcessMessenger::GateVProcessMessenger(GateVProcess *pb):pProcess(pb)
{
  mPrefix = "/gate/physics/";
  //BuildCommands(G4String base)  

  pSetSplit              = 0;
  pSetRussianR           = 0;
  pSetCSE                = 0;
  pAddFilter             = 0;
  pFilteredParticleState = 0;

  pSetEmin        = 0;
  pSetEmax        = 0;
  pClearERange    = 0;

  pAddDataSet     = 0;
  pRemoveDataSet  = 0;
  pListDataSet    = 0;

  pAddModel       = 0;
  pRemoveModel    = 0;
  pListModel      = 0;

  pProcessDir = new G4UIdirectory("/gate/physics/");
  pProcessDir->SetGuidance("GATE physics control.");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateVProcessMessenger::~GateVProcessMessenger()
{
  if(pSetEmin) delete pSetEmin;
  if(pSetEmax) delete pSetEmax;
  if(pClearERange) delete pClearERange;  

  if(pAddDataSet) delete pAddDataSet;
  if(pRemoveDataSet) delete pRemoveDataSet;
  if(pListDataSet) delete pListDataSet;

  if(pAddModel) delete pAddModel;
  if(pRemoveModel) delete pRemoveModel;
  if(pListModel) delete pListModel;

  for (std::vector<GateUIcmdWithADoubleAnd3String *>::iterator it = plModelSetEmin.begin(); it != plModelSetEmin.end(); )
  {
    delete (*it);
    it = plModelSetEmin.erase(it);
  }

  for (std::vector<GateUIcmdWithADoubleAnd3String *>::iterator it = plModelSetEmax.begin(); it != plModelSetEmax.end(); )
  {
    delete (*it);
    it = plModelSetEmax.erase(it);
  }

  for (std::vector<G4UIcmdWithAString *>::iterator it = plModelClearERange.begin(); it != plModelClearERange.end(); )
  {
    delete (*it);
    it = plModelClearERange.erase(it);
  }

  if(pSetSplit) delete pSetSplit;
  if(pSetRussianR) delete pSetRussianR;
  if(pAddFilter) delete pAddFilter;
  if(pSetCSE) delete pSetCSE;
  if(pFilteredParticleState) delete pFilteredParticleState;
  if(pProcessDir) delete pProcessDir;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVProcessMessenger::BuildWrapperCommands(G4String base)
{
  G4String bb;
  G4String guidance;

  base = mPrefix+base;

  bb = base+"/activateSplitting";
  pSetSplit = new GateUIcmdWithAStringAndAnInteger(bb,this);
  guidance = "Activate splitting";
  pSetSplit->SetGuidance(guidance);
  pSetSplit->SetParameterName("Particle","Splitting Factor",false,false);

  bb = base+"/activateRussianRoulette";
  pSetRussianR = new GateUIcmdWithAStringAndAnInteger(bb,this);
  guidance = "Activate splitting";
  pSetRussianR->SetGuidance(guidance);
  pSetRussianR->SetParameterName("Particle","Reduction factor",false,false);

  bb = base+"/activateCSEnhancement";
  pSetCSE = new GateUIcmdWithAStringAndADouble(bb,this);
  guidance = "Activate cross-section enhancement";
  pSetCSE->SetGuidance(guidance);
  pSetCSE->SetParameterName("Particle","factor",false,false);

  bb = base+"/addFilter";
  pAddFilter = new GateUIcmdWith2String(bb,this);
  guidance = "Add filter";
  pAddFilter->SetGuidance(guidance);
  pAddFilter->SetParameterName("Filter type","particles",false,true);
  pAddFilter->SetDefaultValue("","primaries");
  pAddFilter->SetCandidates("","primaries secondaries");

  bb = base+"/keepFilteredSecondaries";
  pFilteredParticleState = new G4UIcmdWithABool(bb,this);
  guidance = "Use 'true' to keep secondary particles filtered or 'false' to kill them.";
  pFilteredParticleState->SetGuidance(guidance);
  pFilteredParticleState->SetParameterName("Keep",false,false);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVProcessMessenger::SetWrapperNewValue(G4UIcommand* command, G4String param)
{

  if(( command==pSetSplit || command==pSetRussianR) && pProcess->GetIsWrapperActive())
    GateError("Splitting or Russian Roulette already activate for the process "<<pProcess->GetG4ProcessName()  <<".");

  if(command==pSetSplit)
  {
     char par1[30];
     int  par2;
     std::istringstream is(param);
     is >> par1 >> par2 ;
     pProcess->SetIsWrapperActive(true);
     pProcess->SetWrapperFactor(par1,par2);
  }
  if(command==pSetRussianR){
     char par1[30];
     int  par2;
     std::istringstream is(param);
     is >> par1 >> par2 ;
     pProcess->SetIsWrapperActive(true);
     pProcess->SetWrapperFactor(par1,1./par2);
  }
  if(command==pSetRussianR){
     char par1[30];
     int  par2;
     std::istringstream is(param);
     is >> par1 >> par2 ;
     pProcess->SetIsWrapperActive(true);
     pProcess->SetWrapperFactor(par1,1./par2);
  }
  if(command==pSetCSE)
  {
     char par1[30];
     double  par2;
     std::istringstream is(param);
     is >> par1 >> par2 ;
     //pProcess->SetIsCSEWrapperActive(true);
     pProcess->SetWrapperCSEFactor(par1,par2);
  }

  if(command==pFilteredParticleState)
  {
    pProcess->SetKeepFilteredSec(pFilteredParticleState->GetNewBoolValue(param));
  }

  if(command==pAddFilter)
  {
      char par1[30];
      char par2[30];
      std::istringstream is(param);
      is >> par1 >> par2 ;
      pProcess->AddFilter(par1, par2 );
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcessMessenger::BuildERangeCommands(G4String ) // Not yet used!
{
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcessMessenger::SetERangeNewValue(G4UIcommand* , G4String ) // Not yet used!
{
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcessMessenger::BuildDataSetCommands(G4String base)
{
  G4String bb;
  G4String guidance;

  std::vector<G4String> lcs = pProcess->GetTheListOfDataSets();

  unsigned int nCS = lcs.size();
  if(nCS == 0) return;

  G4String candCS = "";
  for(unsigned int i=0; i<nCS; i++)
    candCS += lcs[i]  + " ";
  
  base = mPrefix+base;

  bb = base+"/setDataSet";
  pAddDataSet = new GateUIcmdWith2String(bb,this);
  guidance = "Set DataSet for "+ pProcess->GetG4ProcessName();
  pAddDataSet->SetGuidance(guidance);
  pAddDataSet->SetParameterName("DataSet name","Particle or Group of particles",false,true);
  pAddDataSet->SetDefaultValue("","Default");
  pAddDataSet->SetCandidates(candCS,"");

  bb = base+"/unSetDataSet";
  pRemoveDataSet = new GateUIcmdWith2String(bb,this);
  guidance = "Unset DataSet for "+ pProcess->GetG4ProcessName();
  pRemoveDataSet->SetGuidance(guidance);
  pRemoveDataSet->SetParameterName("DataSet name","Particle or Group of particles",false,true);
  pRemoveDataSet->SetDefaultValue("","All");
  pRemoveDataSet->SetCandidates(candCS,"");

  bb = base+"/dataSetList";
  pListDataSet = new G4UIcmdWithAString(bb,this);
  guidance = "List of model(s) for "+ pProcess->GetG4ProcessName();
  pListDataSet->SetGuidance(guidance);
  pListDataSet->SetParameterName("Particle or Group of particles",true);
  pListDataSet->SetDefaultValue("Default");

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcessMessenger::SetDataSetNewValue(G4UIcommand* command, G4String param)
{

  std::vector<G4String> lcs = pProcess->GetTheListOfDataSets();
  unsigned int nCS = lcs.size();
  
  if(nCS == 0) return;

  if(command==pAddDataSet)
  {
      char par1[30];
      char par2[30];
      std::istringstream is(param);
      is >> par1 >> par2 ;
      pProcess->SetDataSet(par1,par2);
  }

  if(command==pRemoveDataSet)
  {
      char par1[30];
      char par2[30];
      std::istringstream is(param);
      is >> par1 >> par2 ;
      pProcess->UnSetDataSet(par1,par2);
  }

  if(command==pListDataSet)
  {
      char par1[30];
      std::istringstream is(param);
      is >> par1 ;
      pProcess->DataSetList(par1);
  }



}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcessMessenger::BuildModelsCommands(G4String base)
{
  G4String bb;
  G4String baseModel = mPrefix+base;
  G4String guidance;

  std::vector<G4String> lmod = pProcess->GetTheListOfModels();

  unsigned int nMod = lmod.size();
  if(nMod == 0) return;

  G4String candModel = "";
  for(unsigned int i=0; i<nMod; i++)
    candModel += lmod[i]  + " ";


  bb = baseModel+"/setModel";
  pAddModel = new GateUIcmdWith2String(bb,this);
  guidance = "Set Model for "+ pProcess->GetG4ProcessName();
  pAddModel->SetGuidance(guidance);
  pAddModel->SetParameterName("Model name","Particle or Group of particles",false,true);
  pAddModel->SetDefaultValue("","Default");
  pAddModel->SetCandidates(candModel,"");

  bb = baseModel+"/unSetModel";
  pRemoveModel = new GateUIcmdWith2String(bb,this);
  guidance = "Unset Model for "+ pProcess->GetG4ProcessName();
  pRemoveModel->SetGuidance(guidance);
  pRemoveModel->SetParameterName("Model name","Particle or Group of particles",false,true);
  pRemoveModel->SetDefaultValue("","All");
  pRemoveModel->SetCandidates(candModel,"");

  bb = baseModel+"/modelList";
  pListModel = new G4UIcmdWithAString(bb,this);
  guidance = "List of model(s) for "+ pProcess->GetG4ProcessName();
  pListModel->SetGuidance(guidance);
  pListModel->SetParameterName("Particle or Group of particles",true);
  pListModel->SetDefaultValue("Default");

}


void GateVProcessMessenger::BuildEnergyRangeModelsCommands(G4String base)
{
  G4String bb;
  G4String baseModel = mPrefix+base;
  G4String guidance;

  std::vector<G4String> lmod = pProcess->GetTheListOfModels();

  unsigned int nMod = lmod.size();
  if(nMod == 0) return;

  for(unsigned int i=0; i<nMod; i++)
  {
    bb = baseModel+"/"+lmod[i]+"/clearEnergyRange";
    plModelClearERange.push_back(new G4UIcmdWithAString(bb,this));
    guidance = "Clear energy range";
    plModelClearERange[i]->SetGuidance(guidance);
    plModelClearERange[i]->SetParameterName("Particle or Group of particles",   true);
    plModelClearERange[i]->SetDefaultValue("All");


    bb = baseModel+"/"+lmod[i]+"/setEmin";
    plModelSetEmin.push_back(new GateUIcmdWithADoubleAnd3String(bb,this));
    guidance = "Set model Emin";
    plModelSetEmin[i]->SetGuidance(guidance);
    plModelSetEmin[i]->SetParameterName("Value","Unit","Particle or Group of particles","Option",false, false,true,  true);
    plModelSetEmin[i]->SetDefaultValue("0.0","MeV","Default","NoOption");
    plModelSetEmin[i]->SetCandidates("","eV keV MeV GeV","","");

    bb = baseModel+"/"+lmod[i]+"/setEmax";
    plModelSetEmax.push_back(new GateUIcmdWithADoubleAnd3String(bb,this));
    guidance = "Set model Emax";
    plModelSetEmax[i]->SetGuidance(guidance);
    plModelSetEmax[i]->SetParameterName("Value","Unit","Particle or Group of particles","Option",false, false, true, true);
    plModelSetEmax[i]->SetDefaultValue("0.0","MeV","Default","NoOption");
    plModelSetEmax[i]->SetCandidates("","eV keV MeV GeV","","");
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcessMessenger::SetModelsNewValue(G4UIcommand* command, G4String param)
{
  std::vector<G4String> lmod = pProcess->GetTheListOfModels();
  unsigned int nMod = lmod.size();
  
  if(nMod == 0) return;

  if(command==pAddModel)
  {
      char par1[30];
      char par2[30];
      std::istringstream is(param);
      is >> par1 >> par2 ;
      pProcess->SetModel(par1,par2);
  }

  if(command==pRemoveModel)
  {
      char par1[30];
      char par2[30];
      std::istringstream is(param);
      is >> par1 >> par2 ;
      pProcess->UnSetModel(par1,par2);
  }

  if(command==pListModel)
  {
      char par1[30];
      std::istringstream is(param);
      is >> par1 ;
      pProcess->ModelList(par1);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVProcessMessenger::SetEnergyRangeModelsNewValue(G4UIcommand* command, G4String param)
{
  std::vector<G4String> lmod = pProcess->GetTheListOfModels();
  unsigned int nMod = lmod.size();
  
  if(nMod == 0) return;

  if(command==pAddModel)
  {
      char par1[30];
      char par2[30];
      std::istringstream is(param);
      is >> par1 >> par2 ;
      pProcess->SetModel(par1,par2);
  }

  if(command==pRemoveModel)
  {
      char par1[30];
      char par2[30];
      std::istringstream is(param);
      is >> par1 >> par2 ;
      pProcess->UnSetModel(par1,par2);
  }

  if(command==pListModel)
  {
      char par1[30];
      std::istringstream is(param);
      is >> par1 ;
      pProcess->ModelList(par1);
  }

  for(unsigned int i=0; i<nMod; i++)
  {
    if(command==plModelClearERange[i])
    {
      char par1[30];
      std::istringstream is(param);
      is >> par1 ;

      pProcess->ClearModelEnergyRange(lmod[i],par1);
    }

    if(command==plModelSetEmin[i])
    {
      double par1;
      char par2[30];
      char par3[30];
      char par4[30];
      std::istringstream is(param);
      is >> par1 >> par2 >> par3 >> par4;
      
      double par1unit = ScaleValue(par1,par2);

      pProcess->SetModelEnergyMin(lmod[i],par1unit,par3,par4);
    }  

    if(command == plModelSetEmax[i])
    { 
      double par1;
      char par2[30];
      char par3[30];
      char par4[30];
      std::istringstream is(param);
      is >> par1 >> par2 >> par3 >> par4 ;
      
      double par1unit = ScaleValue(par1,par2);

      pProcess->SetModelEnergyMax(lmod[i],par1unit,par3,par4);  
    }
  }
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
G4double GateVProcessMessenger::ScaleValue(G4double value,G4String unit)
{
  double res = 0.;
  if(unit=="eV")  res = value *  eV;
  if(unit=="keV") res = value * keV;
  if(unit=="MeV") res = value * MeV;
  if(unit=="GeV") res = value * GeV;

  return res;
}
//-----------------------------------------------------------------------------


#endif /* end #define GATEPROCESSBUILDERMESSENGER_CC */
