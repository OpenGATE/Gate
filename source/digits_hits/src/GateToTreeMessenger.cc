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
  m_addFileNameCmd = new G4UIcmdWithAString( cmdName, this );
  m_addFileNameCmd->SetGuidance(
      "Add name of the output tree data files, extension will set tye type" );
  m_addFileNameCmd->SetParameterName( "Name", false );

//  auto hits_directory = new G4UIdirectory("/gate/output/tree/hits");


  m_enableHitsOutput = new G4UIcmdWithoutParameter("/gate/output/tree/hits/enable", this);
  m_disableHitsOutput = new G4UIcmdWithoutParameter("/gate/output/tree/hits/disable", this);

  m_enableOpticalDataOutput = new G4UIcmdWithoutParameter("/gate/output/tree/optical/enable", this);
  m_disableOpticalDataOutput = new G4UIcmdWithoutParameter("/gate/output/tree/optical/disable", this);

  cmdName = GetDirectoryName() + "addCollection";
  m_addCollectionCmd = new G4UIcmdWithAString(cmdName, this);

  for(auto &&m: m_gateToTree->getHitsParamsToWrite())
  {
    auto name = m.first;
    G4String s = "/gate/output/tree/hits/branches/" + name + "/disable";
    auto c = new G4UIcmdWithoutParameter(s, this);
    m_maphits_cmdParameter_toTreeParameter.emplace(c, name);
  }

  for(auto &&m: m_gateToTree->getOpticalParamsToWrite())
  {
    auto name = m.first;
    G4String s = "/gate/output/tree/optical/branches/" + name + "/disable";
    auto c = new G4UIcmdWithoutParameter(s, this);
    m_mapoptical_cmdParameter_toTreeParameter.emplace(c, name);
  }


  for(auto &&m: m_gateToTree->getSinglesParamsToWrite())
  {
    auto name = m.first;
    G4String s = "/gate/output/tree/Singles/branches/" + name + "/disable";
    auto c = new G4UIcmdWithoutParameter(s, this);
    m_mapsingles_cmdParameter_toTreeParameter.emplace(c, name);
  }

  for(auto &&m: m_gateToTree->getCoincidencesParamsToWrite())
  {
    auto name = m.first;
    G4String s = "/gate/output/tree/Coincidences/branches/" + name + "/disable";
    auto c = new G4UIcmdWithoutParameter(s, this);
    m_mapscoincidences_cmdParameter_toTreeParameter.emplace(c, name);
  }




}

GateToTreeMessenger::~GateToTreeMessenger()
{
  delete m_addFileNameCmd;
  delete m_enableHitsOutput;
  delete m_disableHitsOutput;

}

void GateToTreeMessenger::SetNewValue(G4UIcommand *icommand, G4String string)
{
  GateOutputModuleMessenger::SetNewValue(icommand, string);

  if(icommand == m_addFileNameCmd)
  {
    m_gateToTree->addFileName(string);
  }
  if(icommand == m_enableHitsOutput)
    m_gateToTree->setHitsEnabled(true);
  if(icommand == m_disableHitsOutput)
    m_gateToTree->setHitsEnabled(false);

  if(icommand == m_enableOpticalDataOutput)
    m_gateToTree->setOpticalDataEnabled(true);
  if(icommand == m_disableOpticalDataOutput)
    m_gateToTree->setOpticalDataEnabled(false);



  if(icommand == m_addCollectionCmd)
    m_gateToTree->addCollection(string);

  auto c = static_cast<G4UIcmdWithoutParameter*>(icommand);
  if(m_maphits_cmdParameter_toTreeParameter.count(c))
  {
    auto p = m_maphits_cmdParameter_toTreeParameter.at(c);
    auto &param = m_gateToTree->getHitsParamsToWrite().at(p);
    param.setToSave(false);
  }

  c = static_cast<G4UIcmdWithoutParameter*>(icommand);
  if(m_mapoptical_cmdParameter_toTreeParameter.count(c))
  {
    auto p = m_mapoptical_cmdParameter_toTreeParameter.at(c);
    auto &param = m_gateToTree->getOpticalParamsToWrite().at(p);
    param.setToSave(false);
  }

  c = static_cast<G4UIcmdWithoutParameter*>(icommand);
  if(m_mapsingles_cmdParameter_toTreeParameter.count(c))
  {
    auto p = m_mapsingles_cmdParameter_toTreeParameter.at(c);
    auto &param = m_gateToTree->getSinglesParamsToWrite().at(p);
    param.setToSave(false);
  }

  c = static_cast<G4UIcmdWithoutParameter*>(icommand);
  if(m_mapscoincidences_cmdParameter_toTreeParameter.count(c))
  {
    auto p = m_mapscoincidences_cmdParameter_toTreeParameter.at(c);
    auto &param = m_gateToTree->getCoincidencesParamsToWrite().at(p);
    param.setToSave(false);
  }



}
