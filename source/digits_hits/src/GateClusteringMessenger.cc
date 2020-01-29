
#include "GateClusteringMessenger.hh"

#include "GateClustering.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"


GateClusteringMessenger::GateClusteringMessenger(GateClustering* itsClustering)
    : GatePulseProcessorMessenger(itsClustering)
{


    G4String guidance;
    G4String cmdName;
    G4String cmdName2;

    cmdName = GetDirectoryName() + "setAcceptedDistance";
    pAcceptedDistCmd=new G4UIcmdWithADoubleAndUnit(cmdName,this);
    pAcceptedDistCmd->SetGuidance("Set accepted  distance  for a hit to the center of a cluster to be part of it");
    pAcceptedDistCmd->SetUnitCategory("Length");

    cmdName2 = GetDirectoryName() + "setRejectionMultipleClusters";
    pRejectionMultipleClustersCmd = new  G4UIcmdWithABool(cmdName2,this);
    pRejectionMultipleClustersCmd->SetGuidance("Set to 1 to reject multiple clusters in the same volume");



}


GateClusteringMessenger::~GateClusteringMessenger()
{

    delete pAcceptedDistCmd;
    delete pRejectionMultipleClustersCmd;

}

void GateClusteringMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{


    if ( command==pAcceptedDistCmd)
    {  GetClustering()->SetAcceptedDistance(pAcceptedDistCmd->GetNewDoubleValue(newValue)); }
    else if ( command==pRejectionMultipleClustersCmd )
    { GetClustering()->SetRejectionFlag(pRejectionMultipleClustersCmd->GetNewBoolValue(newValue)); }
    else
        GatePulseProcessorMessenger::SetNewValue(command,newValue);



}


