/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GATEEmCalculatorActorMESSENGER_CC
#define GATEEmCalculatorActorMESSENGER_CC

#include "GateEmCalculatorActorMessenger.hh"
#include "GateEmCalculatorActor.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"



//-----------------------------------------------------------------------------
GateEmCalculatorActorMessenger::GateEmCalculatorActorMessenger(GateEmCalculatorActor* sensor):GateActorMessenger(sensor),pEmCalculatorActor(sensor)
{
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateEmCalculatorActorMessenger::~GateEmCalculatorActorMessenger()
{
  delete pSetEnergyCmd;
  delete pSetParticleNameCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEmCalculatorActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  bb = base+"/setEnergy";
  pSetEnergyCmd = new G4UIcmdWithADoubleAndUnit(bb,this);
  guidance = "Set the energy for the calculation.";
  pSetEnergyCmd->SetGuidance(guidance);

  bb = base+"/setParticleName";
  pSetParticleNameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the particle name for the calculation. For ions, choose 'GenericIon' and specify Z and A with 'setIonProperties'.";
  pSetParticleNameCmd->SetGuidance(guidance);

  //Particle Properties If GenericIon
  bb = base+"/setIonProperties";
  pIonCmd = new G4UIcommand(bb,this);
  // pIonCmd->SetGuidance("Set properties of ion to be generated (if particle is 'GenericIon'):  Z:(int) AtomicNumber, A:(int) AtomicMass, Q:(int) Charge of Ion (in unit of e), E:(double) Excitation energy (in keV).");
  pIonCmd->SetGuidance("Set properties of ion to be generated (if particle is 'GenericIon'):  Z:(int) AtomicNumber, A:(int) AtomicMass, and optionally Q:(int) Charge of Ion (in unit of e), E:(double) Excitation energy (in keV). E.g. for a carbon-12 ion you give '6 12' and for He3 '2 3'.");
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
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEmCalculatorActorMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == pSetEnergyCmd) pEmCalculatorActor->SetEnergy(pSetEnergyCmd->GetNewDoubleValue(param));
  if(command == pSetParticleNameCmd) pEmCalculatorActor->SetParticleName(param);
  if (command == pIonCmd) {
    pEmCalculatorActor->SetIonParameter(param);
    pEmCalculatorActor->SetIsGenericIon(true);
  }

  GateActorMessenger::SetNewValue(command ,param );
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEEmCalculatorActorMESSENGER_CC */
