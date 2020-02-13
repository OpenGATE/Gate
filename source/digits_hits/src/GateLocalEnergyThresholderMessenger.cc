/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateLocalEnergyThresholderMessenger.hh"

#include "GateLocalEnergyThresholder.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

GateLocalEnergyThresholderMessenger::GateLocalEnergyThresholderMessenger(GateLocalEnergyThresholder* itsEnergyThresholder)
    : GatePulseProcessorMessenger(itsEnergyThresholder)
{
  G4String guidance;
    m_count=0;
  G4String cmdName;



   cmdName = GetDirectoryName() + "chooseNewVolume";
    newVolCmd = new G4UIcmdWithAString(cmdName,this);
    newVolCmd->SetGuidance("Choose a volume for threshold");




}

void GateLocalEnergyThresholderMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==newVolCmd )
    {
      G4String cmdName2, cmdName3;

      if(GetLocalEnergyThresholder()->ChooseVolume(newValue) == 1) {
          m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
          m_volDirectory[m_count]->SetGuidance((G4String("characteristics of ") + newValue).c_str());

          m_name.push_back(newValue);

          cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setThreshold";
          thresholdCmd.push_back(new G4UIcmdWithADoubleAndUnit(cmdName2,this));
          thresholdCmd[m_count]->SetGuidance("Set threshold (in keV) for pulse-discrimination");
          thresholdCmd[m_count]->SetUnitCategory("Energy");


          cmdName3 =  m_volDirectory[m_count]->GetCommandPath() + "setLaw";
          lawCmd.push_back(new G4UIcmdWithAString(cmdName3,this));
          lawCmd[m_count]->SetGuidance("Set the law of effective energy  for the threshold");


          m_count++;
      }
    }
  else
    SetNewValue2(command,newValue);
}


void GateLocalEnergyThresholderMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{

    G4int test=0;
      for (G4int i=0;i<m_count;i++)  {
        if ( command==thresholdCmd[i] ) {
          GetLocalEnergyThresholder()->SetThreshold(m_name[i], thresholdCmd[i]->GetNewDoubleValue(newValue));
          test=1;
        }
      }
      if(test==0)
        for (G4int i=0;i<m_count;i++)  {
          if ( command==lawCmd[i] ) {
             GateVEffectiveEnergyLaw* a_energyLaw = CreateEffectiveEnergyLaw( newValue, i);
                if (a_energyLaw != NULL) {
                    GetLocalEnergyThresholder()->SetEffectiveEnergyLaw(m_name[i],a_energyLaw);
                }

        test=1;

          }
        }
      if(test==0)

    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}



GateVEffectiveEnergyLaw* GateLocalEnergyThresholderMessenger::CreateEffectiveEnergyLaw(const G4String& law, int i) {

    if ( law == "solidAngleWeighted" ) {
        G4cout<<"[GateLocalEnergyThresholderMessenger::CreateEffectiveEnergyLaw]"<<m_volDirectory[i]->GetCommandPath()<<G4endl;
        G4cout<<"[GateLocalEnergyThresholderMessenger::CreateEffectiveEnergyLaw]"<<GetLocalEnergyThresholder()->GetObjectName() <<G4endl;
       G4String objectN=m_volDirectory[i]->GetCommandPath();
       std::size_t pos = objectN.find("digitizer");      //

        G4cout<<"[GateLocalEnergyThresholderMessenger::CreateEffectiveEnergyLaw]"<<objectN.substr(pos)<<G4endl;
        return new GateSolidAngleWeightedEnergyLaw(objectN.substr(pos)+ G4String("solidAngleWeighted"));

    } else if ( law == "depositedEnergy" ) {
        G4String objectN=m_volDirectory[i]->GetCommandPath();
        std::size_t pos = objectN.find("digitizer");      //
        //return new GateDepositedEnergyLaw(GetLocalEnergyThresholder()->GetObjectName() + G4String("/depositedEnergy"));
        return new GateDepositedEnergyLaw( objectN.substr(pos)+ G4String("depositedEnergy"));
    } else {
        G4cerr << "No match for '" << law << "' effective energy law.\n";
        G4cerr << "Candidates are: solidAngleWeighted  depositedEnergy\n";
    }

    return NULL;
}


GateLocalEnergyThresholderMessenger::~GateLocalEnergyThresholderMessenger()
{

  delete newVolCmd;
    for (G4int i=0;i<m_count;i++) {
        delete lawCmd[i];
        delete thresholdCmd[i];
      }
}
