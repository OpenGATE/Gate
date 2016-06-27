#include "GateAugerDetectorActorMessenger.hh"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateAugerDetectorActor.hh"

GateAugerDetectorActorMessenger::GateAugerDetectorActorMessenger(GateAugerDetectorActor * v)
: GateActorMessenger(v), pActor(v)
{
	BuildCommands(baseName+pActor->GetObjectName());
}

GateAugerDetectorActorMessenger::~GateAugerDetectorActorMessenger()
{
    delete pMaxTOFCmd;
    delete pMinEdepCmd;
    delete pMaxEdepCmd;
	delete pProfileDirectionCmd;
	delete pMinProfileCmd;
	delete pMaxProfileCmd;
	delete pSizeProfileCmd;
	delete pProfileNoiseFWHMCmd;
}

void GateAugerDetectorActorMessenger::BuildCommands(G4String base)
{
    pMaxTOFCmd = new G4UIcmdWithADoubleAndUnit((base+"/setMinTOF").c_str(),this);
    pMaxTOFCmd->SetGuidance("Set minimum time of flight window");
    pMaxTOFCmd->SetParameterName("MinTOF",false);
    pMaxTOFCmd->SetDefaultUnit("ns");

    pMaxTOFCmd = new G4UIcmdWithADoubleAndUnit((base+"/setMaxTOF").c_str(),this);
    pMaxTOFCmd->SetGuidance("Set maximum time of flight window");
    pMaxTOFCmd->SetParameterName("MaxTOF",false);
    pMaxTOFCmd->SetDefaultUnit("ns");

    pMinEdepCmd = new G4UIcmdWithADoubleAndUnit((base+"/setMinEdep").c_str(),this);
    pMinEdepCmd->SetGuidance("Set minimum energy deposition to trigger detection");
    pMinEdepCmd->SetParameterName("MinEdep",false);
    pMinEdepCmd->SetDefaultUnit("MeV");

    pMaxEdepCmd = new G4UIcmdWithADoubleAndUnit((base+"/setMaxEdep").c_str(),this);
    pMaxEdepCmd->SetGuidance("Set maximum energy deposition to trigger detection");
    pMaxEdepCmd->SetParameterName("MaxEdep",false);
    pMaxEdepCmd->SetDefaultUnit("MeV");

	pProfileDirectionCmd = new G4UIcmdWith3Vector((base+"/setProjectionDirection").c_str(),this);
	pProfileDirectionCmd->SetGuidance("Set the direction in which the profile is built");
	pProfileDirectionCmd->SetParameterName("DirX","DirY","DirZ",false);

	pMinProfileCmd = new G4UIcmdWithADoubleAndUnit((base+"/setProfileMinimum").c_str(),this);
	pMinProfileCmd->SetGuidance("Set the minimum of the profile axis");
	pMinProfileCmd->SetParameterName("ProfileMin",false);
	pMinProfileCmd->SetDefaultUnit("mm");

	pMaxProfileCmd = new G4UIcmdWithADoubleAndUnit((base+"/setProfileMaximum").c_str(),this);
	pMaxProfileCmd->SetGuidance("Set the maximum of the profile axis");
	pMaxProfileCmd->SetParameterName("ProfileMax",false);
	pMaxProfileCmd->SetDefaultUnit("mm");

	pSizeProfileCmd = new G4UIcmdWithAnInteger((base+"/setProfileSize").c_str(),this);
	pSizeProfileCmd->SetGuidance("Set the number of bins (!) in the profile");
	pSizeProfileCmd->SetParameterName("ProfileSize",false);

	pProfileNoiseFWHMCmd = new G4UIcmdWithADoubleAndUnit((base+"/setProfileNoiseFWHM").c_str(),this);
	pProfileNoiseFWHMCmd->SetGuidance("Set profile additive noise FWHM");
	pProfileNoiseFWHMCmd->SetParameterName("ProfileNoiseFWHM",false);
	pProfileNoiseFWHMCmd->SetDefaultUnit("mm");
}

void GateAugerDetectorActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
    if(cmd == pMinTOFCmd) pActor->setMinTOF(  G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue)  );
    if(cmd == pMaxTOFCmd) pActor->setMaxTOF(  G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue)  );
    if(cmd == pMinEdepCmd) pActor->setMinEdep(  G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue)  );
    if(cmd == pMaxEdepCmd) pActor->setMaxEdep(  G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue)  );
	if(cmd == pProfileDirectionCmd) pActor->setProjectionDirection(  G4UIcmdWith3Vector::GetNew3VectorValue(newValue)  );
	if(cmd == pMinProfileCmd) pActor->setMinimumProfileAxis(  G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue)  );
	if(cmd == pMaxProfileCmd) pActor->setMaximumProfileAxis(  G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue)  );
	if(cmd == pSizeProfileCmd) pActor->setProfileSize(  G4UIcmdWithAnInteger::GetNewIntValue(newValue)  );
	if(cmd == pProfileNoiseFWHMCmd) pActor->setProfileNoiseFWHM(  G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue)  );

	GateActorMessenger::SetNewValue(cmd,newValue);
}

#endif
