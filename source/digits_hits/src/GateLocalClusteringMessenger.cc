
#include "GateLocalClusteringMessenger.hh"

#include "GateLocalClustering.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"


GateLocalClusteringMessenger::GateLocalClusteringMessenger(GateLocalClustering* itsPulseAdder)
    : GatePulseProcessorMessenger(itsPulseAdder)
{


      G4String guidance;
      G4String cmdName;
      m_count=0;

      cmdName = GetDirectoryName() + "chooseNewVolume";
      newVolCmd = new G4UIcmdWithAString(cmdName,this);
      newVolCmd->SetGuidance("Choose a volume for applying clustering");




}


GateLocalClusteringMessenger::~GateLocalClusteringMessenger()
{
    delete newVolCmd;
    for (G4int i=0;i<m_count;i++) {
        delete   pRejectionMultipleClustersCmd[i];
        delete pAcceptedDist[i];
    }
}

void GateLocalClusteringMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    if ( command==newVolCmd )
        {
          G4String cmdName2, cmdName3;

           if(GetLocalClustering()->ChooseVolume(newValue) == 1) {
               G4cout<<"new Value options for clustering "<< newValue<<G4endl;


               m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
               m_volDirectory[m_count]->SetGuidance((G4String("characteristics of ") + newValue).c_str());

               m_name.push_back(newValue);

               cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setAcceptedDistance";
               pAcceptedDist.push_back(new G4UIcmdWithADoubleAndUnit(cmdName2,this));
               pAcceptedDist[m_count]->SetGuidance("Set accepted  distance  for a hit to the center of a cluster to be part of it");
               pAcceptedDist[m_count]->SetUnitCategory("Length");


               cmdName3 = m_volDirectory[m_count]->GetCommandPath() + "setRejectionMultipleClusters";
               pRejectionMultipleClustersCmd.push_back(new G4UIcmdWithABool(cmdName3,this));
               pRejectionMultipleClustersCmd[m_count]->SetGuidance("Set to 1 the flag for a continuous readout (one cluster per event)");





           m_count++;
           }
    }
    else
        SetNewValue2(command,newValue);


}

void GateLocalClusteringMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{

    G4int test=0;
      for (G4int i=0;i<m_count;i++)  {
        if ( command==pAcceptedDist[i] ) {
            GetLocalClustering()->SetAcceptedDistance(m_name[i],pAcceptedDist[i]->GetNewDoubleValue(newValue));
          test=1;
        }
      }
      if(test==0)
          for (G4int i=0;i<m_count;i++)  {
            if ( command==pRejectionMultipleClustersCmd[i] ) {
          GetLocalClustering()->SetRejectionFlag(m_name[i], pRejectionMultipleClustersCmd[i]->GetNewBoolValue(newValue));
          test=1;
            }
          }

      if(test==0)
          GatePulseProcessorMessenger::SetNewValue(command,newValue);


}
