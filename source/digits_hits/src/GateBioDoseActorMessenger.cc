/*----------------------
Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateBioDoseActorMessenger.hh"
#include "GateBioDoseActor.hh"

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

	n = base + "/setBioDoseImageFilename";
	pImageFilenameCmd = std::make_unique<G4UIcmdWithAString>(n, this);
	pImageFilenameCmd->SetGuidance("For an image for the biological dose as output");

	n = base + "/setCellLine";
	pCellLineCmd = std::make_unique<G4UIcmdWithAString>(n, this);
	pCellLineCmd->SetGuidance("Pick a cell line to irradiate");

	n = base + "/setBioPhysicalModel";
	pBioPhysicalModelCmd = std::make_unique<G4UIcmdWithAString>(n, this);
	pBioPhysicalModelCmd->SetGuidance("Pick a model to work with");

	n = base + "/setWeightSOBP";
	pSOBPWeightCmd = std::make_unique<G4UIcmdWithADouble>(n, this);
	pSOBPWeightCmd->SetGuidance("If passive SOBP, set bragg peak weight");
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateBioDoseActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue) {
	if(cmd == pAlphaRefCmd.get())								pBioDoseActor->SetAlphaRef(pAlphaRefCmd->GetNewDoubleValue(newValue));
	else if(cmd == pBetaRefCmd.get())						pBioDoseActor->SetBetaRef(pBetaRefCmd->GetNewDoubleValue(newValue));
	else if(cmd == pImageFilenameCmd.get())			pBioDoseActor->SetBioDoseImageFilename(newValue);
	else if(cmd == pCellLineCmd.get())          pBioDoseActor->SetCellLine(newValue);
	else if(cmd == pBioPhysicalModelCmd.get())  pBioDoseActor->SetBioPhysicalModel(newValue);
	else if(cmd == pSOBPWeightCmd.get())        pBioDoseActor->SetSOBPWeight(pSOBPWeightCmd->GetNewDoubleValue(newValue));

	GateImageActorMessenger::SetNewValue(cmd, newValue);
}
//-----------------------------------------------------------------------------
