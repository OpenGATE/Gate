#ifndef GATE_SOURCE_DIGITS_HITS_INCLUDE_GATEBIODOSEACTORMESSENGER_HH
#define GATE_SOURCE_DIGITS_HITS_INCLUDE_GATEBIODOSEACTORMESSENGER_HH

/*----------------------
Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
	\class GateBioDoseActorMessenger
	\author Éloïse Salles, Alexis Pereda <alexis.pereda@clermont.in2p3.fr>, Yasmine Ali
*/

#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "GateImageActorMessenger.hh"
#include <G4UIcmdWithAString.hh>

class GateBioDoseActor;

class GateBioDoseActorMessenger: public GateImageActorMessenger {
public:
	GateBioDoseActorMessenger(GateBioDoseActor* sensor);

	void BuildCommands(G4String base) override;
	void SetNewValue(G4UIcommand*, G4String) override;

private:
	GateBioDoseActor* pBioDoseActor;

	std::unique_ptr<G4UIcmdWithADouble> pDoseScaleFactorCmd;
	std::unique_ptr<G4UIcmdWithADouble> pAlphaRefCmd;
	std::unique_ptr<G4UIcmdWithADouble> pBetaRefCmd;
	std::unique_ptr<G4UIcmdWithAString> pCellLineCmd;
	std::unique_ptr<G4UIcmdWithAString> pBioPhysicalModelCmd;
	std::unique_ptr<G4UIcmdWithADouble> pSOBPWeightCmd;

	std::unique_ptr<G4UIcmdWithABool>   pEnableEdepCmd;
	std::unique_ptr<G4UIcmdWithABool>   pEnableDoseCmd;
	std::unique_ptr<G4UIcmdWithABool>   pEnableBioDoseCmd;
	std::unique_ptr<G4UIcmdWithABool>   pEnableAlphaMixCmd;
	std::unique_ptr<G4UIcmdWithABool>   pEnableSqrtBetaMixCmd;
	std::unique_ptr<G4UIcmdWithABool>   pEnableRBECmd;
	std::unique_ptr<G4UIcmdWithABool>   pEnableUncertaintyCmd;
	std::unique_ptr<G4UIcmdWithABool>   pEnableUncertaintyDetailsCmd;
	std::unique_ptr<G4UIcmdWithABool>   pEnableHitEventCountCmd;
};

#endif
