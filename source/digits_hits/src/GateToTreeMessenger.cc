/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


//
// Created by mdupont on 17/05/19.
//

#include "GateToTreeMessenger.hh"
#include "GateToTree.hh"


#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithoutParameter.hh"

GateToTreeMessenger::GateToTreeMessenger(GateToTree *m) :
    GateOutputModuleMessenger(m),
    m_gateToTree(m)
{

  G4String cmdName = GetDirectoryName() + "addFileName";
  m_addFileNameCmd.reset(new G4UIcmdWithAString( cmdName, this ));
  m_addFileNameCmd->SetGuidance(
      "Add name of the output tree data files, extension will set tye type" );
  m_addFileNameCmd->SetParameterName( "Name", false );
  

  m_enableHitsOutput.reset(new G4UIcmdWithoutParameter("/gate/output/tree/hits/enable", this));
  m_disableHitsOutput.reset(new G4UIcmdWithoutParameter("/gate/output/tree/hits/disable", this));
  m_disableAllHitsBranches.reset(new G4UIcmdWithoutParameter("/gate/output/tree/hits/branches/disable", this));


  m_enableOpticalDataOutput.reset(new G4UIcmdWithoutParameter("/gate/output/tree/optical/enable", this));
  m_disableOpticalDataOutput.reset(new G4UIcmdWithoutParameter("/gate/output/tree/optical/disable", this));
  m_disableOpticalDataBranches.reset(new G4UIcmdWithoutParameter("/gate/output/tree/optical/branches/disable", this));

  cmdName = GetDirectoryName() + "addCollection";
  m_addCollectionCmd.reset(new G4UIcmdWithAString(cmdName, this));
  
  m_disableAllSinglesBranches.reset(new G4UIcmdWithoutParameter("/gate/output/tree/Singles/branches/disable", this));
  m_disableAllCoincidencesBranches.reset(new G4UIcmdWithoutParameter("/gate/output/tree/Coincidences/branches/disable", this));

  for(auto &&m: m_gateToTree->getHitsParamsToWrite())
  {
    auto name = m.first;
    G4String s = "/gate/output/tree/hits/branches/" + name + "/disable";
    auto c = new G4UIcmdWithoutParameter(s, this);
    m_maphits_cmdParameter_toTreeParameter_disable.emplace(c, name);
    
    s = "/gate/output/tree/hits/branches/" + name + "/enable";
    c = new G4UIcmdWithoutParameter(s, this);
    m_maphits_cmdParameter_toTreeParameter_enable.emplace(c, name);
  }

  for(auto &&m: m_gateToTree->getOpticalParamsToWrite())
  {
    auto name = m.first;
    G4String s = "/gate/output/tree/optical/branches/" + name + "/disable";
    auto c = new G4UIcmdWithoutParameter(s, this);
    m_mapoptical_cmdParameter_toTreeParameter_disable.emplace(c, name);
    
    s = "/gate/output/tree/optical/branches/" + name + "/enable";
    c = new G4UIcmdWithoutParameter(s, this);
    m_mapoptical_cmdParameter_toTreeParameter_enable.emplace(c, name);
  }


  for(auto &&m: m_gateToTree->getSinglesParamsToWrite())
  {
    auto name = m.first;
    G4String s = "/gate/output/tree/Singles/branches/" + name + "/disable";
    auto c = new G4UIcmdWithoutParameter(s, this);
    m_mapsingles_cmdParameter_toTreeParameter_disable.emplace(c, name);
    
    s = "/gate/output/tree/Singles/branches/" + name + "/enable";
    c = new G4UIcmdWithoutParameter(s, this);
    m_mapsingles_cmdParameter_toTreeParameter_enable.emplace(c, name);
  }

  for(auto &&m: m_gateToTree->getCoincidencesParamsToWrite())
  {
    auto name = m.first;
    G4String s = "/gate/output/tree/Coincidences/branches/" + name + "/disable";
    auto c = new G4UIcmdWithoutParameter(s, this);
    m_mapscoincidences_cmdParameter_toTreeParameter_disable.emplace(c, name);
    
    s = "/gate/output/tree/Coincidences/branches/" + name + "/enable";
    c = new G4UIcmdWithoutParameter(s, this);
    m_mapscoincidences_cmdParameter_toTreeParameter_enable.emplace(c, name);
  }
  
  m_useRootFriendlyFormat.reset(new G4UIcmdWithoutParameter("/gate/output/tree/enableRootFriendlyFormat", this));
}

GateToTreeMessenger::~GateToTreeMessenger()
{
 DeleteMap(m_maphits_cmdParameter_toTreeParameter_disable);
 DeleteMap(m_maphits_cmdParameter_toTreeParameter_enable);
 DeleteMap(m_mapoptical_cmdParameter_toTreeParameter_disable);
 DeleteMap(m_mapoptical_cmdParameter_toTreeParameter_enable);
 DeleteMap(m_mapsingles_cmdParameter_toTreeParameter_disable);
 DeleteMap(m_mapsingles_cmdParameter_toTreeParameter_enable);
 DeleteMap(m_mapscoincidences_cmdParameter_toTreeParameter_disable);
 DeleteMap(m_mapscoincidences_cmdParameter_toTreeParameter_enable);
}

void GateToTreeMessenger::SetNewValue(G4UIcommand *icommand, G4String string)
{
  GateOutputModuleMessenger::SetNewValue(icommand, string);

  if(icommand == m_addFileNameCmd.get())
  {
    m_gateToTree->addFileName(string);
  }
  if(icommand == m_enableHitsOutput.get())
    m_gateToTree->setHitsEnabled(true);
  if(icommand == m_disableHitsOutput.get())
    m_gateToTree->setHitsEnabled(false);
  if(icommand == m_disableAllHitsBranches.get())
    m_gateToTree->setHitsBranchesEnable(false);

  if(icommand == m_enableOpticalDataOutput.get())
    m_gateToTree->setOpticalDataEnabled(true);
  if(icommand == m_disableOpticalDataOutput.get())
    m_gateToTree->setOpticalDataEnabled(false);
  if(icommand == m_disableOpticalDataBranches.get())
    m_gateToTree->setOpticalDataBranchesEnable(false);
    

  if(icommand == m_addCollectionCmd.get())
    m_gateToTree->addCollection(string);
    
  if(icommand == m_disableAllSinglesBranches.get())
   m_gateToTree->setSingleDigiBranchesEnable(false);
   
  if(icommand == m_disableAllCoincidencesBranches.get())
   m_gateToTree->setCoincidenceDigiBranchesEnable(false);
   
  if(icommand == m_useRootFriendlyFormat.get())
   m_gateToTree->setRootFriendlyFormat(true);

  auto c = static_cast<G4UIcmdWithoutParameter*>(icommand);
  if(m_maphits_cmdParameter_toTreeParameter_disable.count(c))
  {
    auto p = m_maphits_cmdParameter_toTreeParameter_disable.at(c);
    auto &param = m_gateToTree->getHitsParamsToWrite().at(p);
    param.setToSave(false);
  }
  else if (m_maphits_cmdParameter_toTreeParameter_enable.count(c))
  {
    auto p = m_maphits_cmdParameter_toTreeParameter_enable.at(c);
    auto &param = m_gateToTree->getHitsParamsToWrite().at(p);
    param.setToSave(true);  
  }

  c = static_cast<G4UIcmdWithoutParameter*>(icommand);
  if(m_mapoptical_cmdParameter_toTreeParameter_disable.count(c))
  {
    auto p = m_mapoptical_cmdParameter_toTreeParameter_disable.at(c);
    auto &param = m_gateToTree->getOpticalParamsToWrite().at(p);
    param.setToSave(false);
  }
  else if(m_mapoptical_cmdParameter_toTreeParameter_enable.count(c))
  {
    auto p = m_mapoptical_cmdParameter_toTreeParameter_enable.at(c);
    auto &param = m_gateToTree->getOpticalParamsToWrite().at(p);
    param.setToSave(true);
  }

  c = static_cast<G4UIcmdWithoutParameter*>(icommand);
  if(m_mapsingles_cmdParameter_toTreeParameter_disable.count(c))
  {
    auto p = m_mapsingles_cmdParameter_toTreeParameter_disable.at(c);
    auto &param = m_gateToTree->getSinglesParamsToWrite().at(p);
    param.setToSave(false);
  }
  else if(m_mapsingles_cmdParameter_toTreeParameter_enable.count(c))
  {
    auto p = m_mapsingles_cmdParameter_toTreeParameter_enable.at(c);
    auto &param = m_gateToTree->getSinglesParamsToWrite().at(p);
    param.setToSave(true);
  }

  c = static_cast<G4UIcmdWithoutParameter*>(icommand);
  if(m_mapscoincidences_cmdParameter_toTreeParameter_disable.count(c))
  {
    auto p = m_mapscoincidences_cmdParameter_toTreeParameter_disable.at(c);
    auto &param = m_gateToTree->getCoincidencesParamsToWrite().at(p);
    param.setToSave(false);
  }
  else if(m_mapscoincidences_cmdParameter_toTreeParameter_enable.count(c))
  {
    auto p = m_mapscoincidences_cmdParameter_toTreeParameter_enable.at(c);
    auto &param = m_gateToTree->getCoincidencesParamsToWrite().at(p);
    param.setToSave(false);
  }
}

void GateToTreeMessenger::DeleteMap( std::unordered_map<G4UIcmdWithoutParameter*, G4String>& m )
{
 for ( std::unordered_map<G4UIcmdWithoutParameter*, G4String>::iterator it = m.begin(); it != m.end(); ++it )
  delete it->first;
 m.clear();
}
