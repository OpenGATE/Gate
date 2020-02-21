

#include "GateCC3DlocalSpblurringMessenger.hh"
#include "GateCC3DlocalSpblurring.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

GateCC3DlocalSpblurringMessenger::GateCC3DlocalSpblurringMessenger(GateCC3DlocalSpblurring* itsDelay)
    : GatePulseProcessorMessenger(itsDelay)
{
    G4String guidance;
    G4String cmdName;
    m_count=0;

    cmdName = GetDirectoryName() + "chooseNewVolume";
    newVolCmd = new G4UIcmdWithAString(cmdName,this);
    newVolCmd->SetGuidance("Choose a volume for  TimeDelay");
}


GateCC3DlocalSpblurringMessenger::~GateCC3DlocalSpblurringMessenger()
{
    delete newVolCmd;
    for (G4int i=0;i<m_count;i++) {
        delete sigmaCmd[i];
    }
}


void GateCC3DlocalSpblurringMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    if ( command==newVolCmd )
    {
        G4String cmdName2;

        if(GetLocal3DSpBlurring()->ChooseVolume(newValue) == 1) {
            m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
            m_volDirectory[m_count]->SetGuidance((G4String("characteristics of ") + newValue).c_str());
            m_name.push_back(newValue);

            cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setSigma";
            sigmaCmd.push_back( new G4UIcmdWith3VectorAndUnit(cmdName2,this));
            sigmaCmd[m_count]->SetGuidance("Set the sigma for the spatial gaussian blurring in each direction.");
            sigmaCmd[m_count]->SetParameterName("SgmX","SgmY","SgmZ",false);
            sigmaCmd[m_count]->SetUnitCategory("Length");

            m_count++;
        }
    }
    else
        SetNewValue2(command,newValue);
}

void GateCC3DlocalSpblurringMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{
    G4int test=0;
    for (G4int i=0;i<m_count;i++)  {
        if ( command==sigmaCmd[i] ) {
            GetLocal3DSpBlurring()->SetSigmaSpBlurring(m_name[i], sigmaCmd[i]->GetNew3VectorValue(newValue));
            test=1;
        }
    }

    if(test==0)
        GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
