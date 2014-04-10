
#include "GateGpTLEActorMessenger.hh"

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateGpTLEActor.hh"

GateGpTLEActorMessenger::GateGpTLEActorMessenger(GateGpTLEActor* v)
: GateActorMessenger(v), pGpTLEActor(v)
{

	BuildCommands(baseName+pActor->GetObjectName());

}

GateGpTLEActorMessenger::~GateGpTLEActorMessenger()
{
	//delete pEmaxCmd;
	//delete pEminCmd;
        delete pFileSpectreBaseNameCmd;
        delete pSaveFilenameCmd;
}

void GateGpTLEActorMessenger::BuildCommands( G4String base)
{
	//bb = base+"/energySpectrum/setEmin";
	//pEminCmd = new G4UIcmdWithADoubleAndUnit(bb, this); 
	//guidance = G4String("Set minimum energy of the energy spectrum");
	//pEminCmd->SetGuidance(guidance);
	//pEminCmd->SetParameterName("Emin", false);
	//pEminCmd->SetDefaultUnit("MeV");

	//bb = base+"/energyLossHisto/setNumberOfBins";
	//pEdepNBinsCmd = new G4UIcmdWithAnInteger(bb, this); 
	//guidance = G4String("Set number of bins of the energy loss histogram");
	//pEdepNBinsCmd->SetGuidance(guidance);
	//pEdepNBinsCmd->SetParameterName("Nbins", false);

        G4String n, guid;
        n = base+"/FileSpectreBaseName";
        pFileSpectreBaseNameCmd = new G4UIcmdWithAString(n, this);
        guid = G4String( "Base Name of Spectre files");
        pFileSpectreBaseNameCmd->SetGuidance( guid);

        n = base+"/SaveFilename";
        pSaveFilenameCmd = new G4UIcmdWithAString(n, this);
        guid = G4String( "Name of Spectrum out file");
        pSaveFilenameCmd->SetGuidance( guid);
}

void GateGpTLEActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
	//if(cmd == pEminCmd) pActor->SetEmin(  pEminCmd->GetNewDoubleValue(newValue)  ) ;


        if( cmd == pFileSpectreBaseNameCmd)
          {
            pGpTLEActor->FileSpectreBaseName(newValue);
          }

        if( cmd == pSaveFilenameCmd)
          {
            pGpTLEActor->SaveFilename(newValue);
          }

        GateActorMessenger::SetNewValue(cmd,newValue);
}

#endif
