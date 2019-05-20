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

  auto hits_directory = new G4UIdirectory("/gate/output/tree/hits");


  m_enableHitsOutput = new G4UIcmdWithoutParameter("/gate/output/tree/hits/enable", this);
  m_disableHitsOutput = new G4UIcmdWithoutParameter("/gate/output/tree/hits/disable", this);

  cmdName = GetDirectoryName() + "addCollection";
  m_addCollectionCmd = new G4UIcmdWithAString(cmdName, this);




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
  if(icommand == m_addCollectionCmd)
    m_gateToTree->addCollection(string);



}
