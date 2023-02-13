/*----------------------
Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateBioDoseActorMessenger.hh"
#include "GateBioDoseActor.hh"
#include <G4UIcmdWithABool.hh>
#include <memory>

//-----------------------------------------------------------------------------
GateBioDoseActorMessenger::GateBioDoseActorMessenger(GateBioDoseActor* sensor):
	GateImageActorMessenger(sensor),
	pBioDoseActor(sensor)
{
	BuildCommands(baseName + sensor->GetObjectName());
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActorMessenger::BuildCommands(G4String base) {
	G4String n;

	n = base + "/setAlphaRef";
	pAlphaRefCmd = std::make_unique<G4UIcmdWithADouble>(n, this);
	pAlphaRefCmd->SetGuidance("See [...] for values from publications");

	n = base + "/setBetaRef";
	pBetaRefCmd = std::make_unique<G4UIcmdWithADouble>(n, this);
	pBetaRefCmd->SetGuidance("See [...] for values from publications");

	n = base + "/setCellLine";
	pCellLineCmd = std::make_unique<G4UIcmdWithAString>(n, this);
	pCellLineCmd->SetGuidance("Pick a cell line to irradiate");

	n = base + "/setBioPhysicalModel";
	pBioPhysicalModelCmd = std::make_unique<G4UIcmdWithAString>(n, this);
	pBioPhysicalModelCmd->SetGuidance("Pick a model to work with");

	n = base + "/setWeightSOBP";
	pSOBPWeightCmd = std::make_unique<G4UIcmdWithADouble>(n, this);
	pSOBPWeightCmd->SetGuidance("If passive SOBP, set bragg peak weight");

	// enable outputs
	n = base + "/enableDose";
	pEnableDoseCmd = std::make_unique<G4UIcmdWithABool>(n, this);
	pEnableDoseCmd->SetGuidance("Enable dose output");

	n = base + "/enableBioDose";
	pEnableBioDoseCmd = std::make_unique<G4UIcmdWithABool>(n, this);
	pEnableBioDoseCmd->SetGuidance("Enable biodose output (default: true)");

	n = base + "/enableAlphaMix";
	pEnableAlphaMixCmd = std::make_unique<G4UIcmdWithABool>(n, this);
	pEnableAlphaMixCmd->SetGuidance("Enable alpha mix output");

	n = base + "/enableBetaMix";
	pEnableBetaMixCmd = std::make_unique<G4UIcmdWithABool>(n, this);
	pEnableBetaMixCmd->SetGuidance("Enable beta mix output");

	n = base + "/enableRBE";
	pEnableRBECmd = std::make_unique<G4UIcmdWithABool>(n, this);
	pEnableRBECmd->SetGuidance("Enable RBE output");
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActorMessenger::SetNewValue(G4UIcommand* cmd, G4String value) {
	if(cmd == pAlphaRefCmd.get())								pBioDoseActor->SetAlphaRef(pAlphaRefCmd->GetNewDoubleValue(value));
	else if(cmd == pBetaRefCmd.get())						pBioDoseActor->SetBetaRef(pBetaRefCmd->GetNewDoubleValue(value));
	else if(cmd == pCellLineCmd.get())          pBioDoseActor->SetCellLine(value);
	else if(cmd == pBioPhysicalModelCmd.get())  pBioDoseActor->SetBioPhysicalModel(value);
	else if(cmd == pSOBPWeightCmd.get())        pBioDoseActor->SetSOBPWeight(pSOBPWeightCmd->GetNewDoubleValue(value));
	else if(cmd == pEnableDoseCmd.get())        pBioDoseActor->SetEnableDose(pEnableDoseCmd->GetNewBoolValue(value));
	else if(cmd == pEnableBioDoseCmd.get())     pBioDoseActor->SetEnableBioDose(pEnableBioDoseCmd->GetNewBoolValue(value));
	else if(cmd == pEnableAlphaMixCmd.get())    pBioDoseActor->SetEnableAlphaMix(pEnableAlphaMixCmd->GetNewBoolValue(value));
	else if(cmd == pEnableBetaMixCmd.get())     pBioDoseActor->SetEnableBetaMix(pEnableBetaMixCmd->GetNewBoolValue(value));
	else if(cmd == pEnableRBECmd.get())         pBioDoseActor->SetEnableRBE(pEnableRBECmd->GetNewBoolValue(value));

	GateImageActorMessenger::SetNewValue(cmd, value);
}
//-----------------------------------------------------------------------------
