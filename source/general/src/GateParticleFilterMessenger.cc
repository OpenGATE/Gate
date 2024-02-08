/*----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GATEPARTFILTERMESSENGER_CC
#define GATEPARTFILTERMESSENGER_CC

#include "GateParticleFilterMessenger.hh"

#include "GateParticleFilter.hh"

#include "GateConfiguration.h"

//-----------------------------------------------------------------------------
GateParticleFilterMessenger::GateParticleFilterMessenger(GateParticleFilter* partFilter)

  : pParticleFilter(partFilter)
{
  BuildCommands(pParticleFilter->GetObjectName());

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateParticleFilterMessenger::~GateParticleFilterMessenger()
{
  delete pAddParticleCmd;
  delete pInvertCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateParticleFilterMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;


  bb = base+"/addParticle";
  pAddParticleCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Add particle";
  pAddParticleCmd->SetGuidance(guidance);
  pAddParticleCmd->SetParameterName("Particle name",false);

  bb = base+"/addParticleZ";
  pAddParticleZCmd = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Add particles by atomic number";
  pAddParticleZCmd->SetGuidance(guidance);
  pAddParticleZCmd->SetParameterName("Particles atomic number",false);

  bb = base+"/addParticleA";
  pAddParticleACmd = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Add particles by atomic mass";
  pAddParticleACmd->SetGuidance(guidance);
  pAddParticleACmd->SetParameterName("Particles atomic mass A",false);

  bb = base+"/addParticlePDG";
  pAddParticlePDGCmd = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Add particles by PDG number";
  pAddParticlePDGCmd->SetGuidance(guidance);
  pAddParticlePDGCmd->SetParameterName("Particles PDG number",false);

  bb = base+"/addParentParticle";
  pAddParentParticleCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Add parent particle";
  pAddParentParticleCmd->SetGuidance(guidance);
  pAddParentParticleCmd->SetParameterName("Particle name",false);

  bb = base+"/addDirectParentParticle";
  pAddDirectParentParticleCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Add direct parent particle";
  pAddDirectParentParticleCmd->SetGuidance(guidance);
  pAddDirectParentParticleCmd->SetParameterName("Particle name",false);

  bb = base+"/invert";
  pInvertCmd = new G4UIcmdWithoutParameter(bb,this);
  guidance = "Invert the filter";
  pInvertCmd->SetGuidance(guidance);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateParticleFilterMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command==pAddParticleCmd)
    pParticleFilter->Add(param);
  if(command==pAddParticleZCmd)
    pParticleFilter->AddZ(pAddParticleZCmd->GetNewIntValue(param));
  if(command==pAddParticleACmd)
    pParticleFilter->AddA(pAddParticleZCmd->GetNewIntValue(param));
  if(command==pAddParticlePDGCmd)
    pParticleFilter->AddPDG(pAddParticleZCmd->GetNewIntValue(param));
  if(command==pAddParentParticleCmd)
    pParticleFilter->AddParent(param);
  if(command==pAddDirectParentParticleCmd)
    pParticleFilter->AddDirectParent(param);
  if(command==pInvertCmd)
    pParticleFilter->setInvert();
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEPARTFILTERMESSENGER_CC */
