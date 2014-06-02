/*##########################################
#developed by Hermann Fuchs
#
#Christian Doppler Laboratory for Medical Radiation Research for Radiation Oncology
#Department of Radiation Oncology
#Medical University of Vienna
#
#and 
#
#Pierre Gueth
#CREATIS
#
#July 2012
##########################################
*/
#include "GateCerenkovMessenger.hh"
#include "GateVProcess.hh"

GateCerenkovMessenger::GateCerenkovMessenger(GateVProcess *pb):GateVProcessMessenger(pb)
{
	maxNumPhotonsPerStep = 300;
	trackSecondariesFirst = false;
	maxBetaChangePerStep = 0;

	BuildCommands("processes/" + pb->GetG4ProcessName());
}

GateCerenkovMessenger::~GateCerenkovMessenger()
{
	delete pSetMaxNumPhotonsPerStep;
	delete pSetTrackSecondariesFirst;
	delete pSetMaxBetaChangePerStep;
}

void GateCerenkovMessenger::BuildCommands(G4String base)
{
	{
		pSetMaxNumPhotonsPerStep = new G4UIcmdWithAnInteger((mPrefix+base+"/setMaxNumPhotonsPerStep").c_str(),this);
		pSetMaxNumPhotonsPerStep->SetParameterName("maxNumPhotonsPerStep", false);
		pSetMaxNumPhotonsPerStep->SetRange("maxNumPhotonsPerStep >= 0");
		pSetMaxNumPhotonsPerStep->SetDefaultValue(maxNumPhotonsPerStep);
		pSetMaxNumPhotonsPerStep->SetGuidance("Set maximum number of photons created per step");
	} {
		pSetTrackSecondariesFirst = new G4UIcmdWithABool((mPrefix+base+"/setTrackSecondariesFirst").c_str(),this);
		pSetTrackSecondariesFirst->SetParameterName("trackSecondariesFirst", false);
		pSetTrackSecondariesFirst->SetDefaultValue(trackSecondariesFirst);
		pSetTrackSecondariesFirst->SetGuidance("Track photons before next step occurs");
	} {
		pSetMaxBetaChangePerStep = new G4UIcmdWithADouble((mPrefix+base+"/setMaxBetaChangePerStep").c_str(),this);
		pSetMaxBetaChangePerStep->SetParameterName("maxBetaChangePerStep", false);
		pSetMaxBetaChangePerStep->SetDefaultValue(maxBetaChangePerStep);
		pSetMaxBetaChangePerStep->SetGuidance("Set the maximum allowed change in beta = v/c in % (perCent) per step");
	}
}

void GateCerenkovMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
	if (command == pSetMaxNumPhotonsPerStep) { maxNumPhotonsPerStep = pSetMaxNumPhotonsPerStep->GetNewIntValue(param); }
	if (command == pSetTrackSecondariesFirst) { trackSecondariesFirst = pSetTrackSecondariesFirst->GetNewBoolValue(param); }
	if (command == pSetMaxBetaChangePerStep) { maxBetaChangePerStep = pSetMaxBetaChangePerStep->GetNewDoubleValue(param); }
}

