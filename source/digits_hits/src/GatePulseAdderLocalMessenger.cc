/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/
//GND 2022 Class to Removex

#include "GatePulseAdderLocalMessenger.hh"

#include "GatePulseAdderLocal.hh"

GatePulseAdderLocalMessenger::GatePulseAdderLocalMessenger(GatePulseAdderLocal* itsPulseAdderLocal)
    : GatePulseProcessorMessenger(itsPulseAdderLocal)
{

    G4String guidance;
    G4String cmdName;

    m_count=0;

    cmdName = GetDirectoryName() + "chooseNewVolume";
    newVolCmd = new G4UIcmdWithAString(cmdName,this);
    newVolCmd->SetGuidance("Choose a volume to apply loacal adder volume ");



}

GatePulseAdderLocalMessenger::~GatePulseAdderLocalMessenger()
{
    delete   newVolCmd;
    for (G4int i=0;i<m_count;i++) {
        delete positionPolicyCmd[i];
    }
}



void GatePulseAdderLocalMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    if ( command==newVolCmd )
    {
        G4String cmdName2;

        if(GetPulseAdderLocal()->chooseVolume(newValue) == 1) {
            m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
            m_volDirectory[m_count]->SetGuidance((G4String("characteristics of ") + newValue).c_str());
            m_name.push_back(newValue);

            cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "positionPolicy";
            positionPolicyCmd.push_back( new G4UIcmdWithAString(cmdName2,this));
            positionPolicyCmd[m_count]->SetGuidance("How to generate position");
            positionPolicyCmd[m_count]->SetCandidates("energyWeightedCentroid takeEnergyWinner");

            m_count++;
        }
    }
    else
        SetNewValue2(command,newValue);
}




void GatePulseAdderLocalMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{
    G4int test=0;
    for (G4int i=0;i<m_count;i++)  {
        if (command==positionPolicyCmd[i] ) {
            GetPulseAdderLocal()->SetPositionPolicy(m_name[i], newValue);
            test=1;
        }
    }

    if(test==0)
        GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
