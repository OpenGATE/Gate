
#include "GateGridDiscretizationMessenger.hh"

#include "GateGridDiscretization.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"


GateGridDiscretizationMessenger::GateGridDiscretizationMessenger(GateGridDiscretization* itsPulseAdder)
    : GatePulseProcessorMessenger(itsPulseAdder)
{


      G4String guidance;
      G4String cmdName;
      m_count=0;

      cmdName = GetDirectoryName() + "chooseNewVolume";

      newVolCmd = new G4UIcmdWithAString(cmdName,this);
      newVolCmd->SetGuidance("Choose a volume for applying a grid discretization");
        G4cout<<"DiscretizationMessenger: "<<cmdName<<Gateendl;

}


GateGridDiscretizationMessenger::~GateGridDiscretizationMessenger()
{
    delete newVolCmd;
    for (G4int i=0;i<m_count;i++) {
        //delete pthresholdCmd[i];
        delete pStripOffsetX[i];
        delete pStripOffsetY[i];
        delete pStripWidthX[i];
        delete pStripWidthY[i];
        delete pNumberStripsX[i];
        delete pNumberStripsY[i];
        delete pNumberReadOutBlocksY[i];
        delete pNumberReadOutBlocksX[i];
        //delete pRejectionMultiplesCmd[i];
    }
}

void GateGridDiscretizationMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    if ( command==newVolCmd )
        {
        G4cout<<"Values for the volume "<< newVolCmd->GetCommandName()<<G4endl;

          G4String cmdName2,cmdName11;
            G4String  cmdName3, cmdName4, cmdName5,cmdName6,cmdName7,cmdName8, cmdName9, cmdName10;

           if(GetGridDiscretization()->ChooseVolume(newValue) == 1) {
               G4cout<<"new Value options for SpDiscreti "<< newValue<<G4endl;


               m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
               m_volDirectory[m_count]->SetGuidance((G4String("characteristics of ") + newValue).c_str());

               m_name.push_back(newValue);

             //  cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setThreshold";
              // pthresholdCmd.push_back(new G4UIcmdWithADoubleAndUnit(cmdName2,this));
              // pthresholdCmd[m_count]->SetGuidance("Set threshold (in keV) for activation of strips");
              // pthresholdCmd[m_count]->SetUnitCategory("Energy");


               cmdName3 = m_volDirectory[m_count]->GetCommandPath() + "setNumberStripsX";
               pNumberStripsX.push_back(new G4UIcmdWithAnInteger(cmdName3,this));
               pNumberStripsX[m_count]->SetGuidance("Set Number of Strips in X direction");


               cmdName4 = m_volDirectory[m_count]->GetCommandPath() + "setNumberStripsY";
               pNumberStripsY.push_back(new G4UIcmdWithAnInteger(cmdName4,this));
               pNumberStripsY[m_count]->SetGuidance("Set Number of Strips in Y direction");


               cmdName5 = m_volDirectory[m_count]->GetCommandPath() + "setStripOffsetX";
               pStripOffsetX.push_back(new G4UIcmdWithADoubleAndUnit(cmdName5,this));
               pStripOffsetX[m_count]->SetGuidance("Set offset of the strip in X direction from negative axis");
               pStripOffsetX[m_count]->SetUnitCategory("Length");

               cmdName6 = m_volDirectory[m_count]->GetCommandPath() + "setStripOffsetY";
               pStripOffsetY.push_back(new G4UIcmdWithADoubleAndUnit(cmdName6,this));
               pStripOffsetX[m_count]->SetGuidance("Set offset of the strip in Y direction from negative axis");
               pStripOffsetY[m_count]->SetUnitCategory("Length");


               cmdName7 = m_volDirectory[m_count]->GetCommandPath() + "setStripWidthX";
               pStripWidthX.push_back(new G4UIcmdWithADoubleAndUnit(cmdName7,this));
               pStripWidthX[m_count]->SetGuidance("Set width of the strip in X direction");
               pStripWidthX[m_count]->SetUnitCategory("Length");

               cmdName8 = m_volDirectory[m_count]->GetCommandPath() + "setStripWidthY";
               pStripWidthY.push_back(new G4UIcmdWithADoubleAndUnit(cmdName8,this));
               pStripWidthY[m_count]->SetGuidance("Set width of the strip in Y direction");
               pStripWidthY[m_count]->SetUnitCategory("Length");


               cmdName9 = m_volDirectory[m_count]->GetCommandPath() + "setNumberReadOutBlocksX";
               pNumberReadOutBlocksX.push_back(new G4UIcmdWithAnInteger(cmdName9,this));
               pNumberReadOutBlocksX[m_count]->SetGuidance("Set Number of ReadOut blocks in X direction");


               cmdName10 = m_volDirectory[m_count]->GetCommandPath() + "setNumberReadOutBlocksY";
               pNumberReadOutBlocksY.push_back(new G4UIcmdWithAnInteger(cmdName10,this));
               pNumberReadOutBlocksY[m_count]->SetGuidance("Set Number of Readout blocks Y direction");


              // cmdName11 = m_volDirectory[m_count]->GetCommandPath() + "setMultipleRejectionflag";
              // pRejectionMultiplesCmd.push_back(new G4UIcmdWithABool(cmdName11,this));
              // pRejectionMultiplesCmd[m_count]->SetGuidance("Set to 1 the flag to reject those events with multiple singles)");



              GetGridDiscretization()->SetVolumeName(newValue);

           m_count++;
           }
    }
    else
        SetNewValue2(command,newValue);


}

void GateGridDiscretizationMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{

    G4int test=0;
   // for (G4int i=0;i<m_count;i++)  {
   //     if ( command==pthresholdCmd[i] ) {
   //         GetGridDiscretization()->SetThreshold(m_name[i],pthresholdCmd[i]->GetNewDoubleValue(newValue));
   //         test=1;
   //    }
   //}
    //if(test==0)
        for (G4int i=0;i<m_count;i++)  {
            if ( command==pStripOffsetX[i] ) {
                GetGridDiscretization()->SetStripOffsetX(m_name[i], pStripOffsetX[i]->GetNewDoubleValue(newValue));
                test=1;
            }
        }
    if(test==0)
        for (G4int i=0;i<m_count;i++)  {
            if ( command==pStripOffsetY[i] ) {
                GetGridDiscretization()->SetStripOffsetY(m_name[i], pStripOffsetY[i]->GetNewDoubleValue(newValue));
                test=1;
            }
        }
    if(test==0)
        for (G4int i=0;i<m_count;i++)  {
            if ( command==pStripWidthX[i] ) {
                GetGridDiscretization()->SetStripWidthX(m_name[i], pStripWidthX[i]->GetNewDoubleValue(newValue));
                test=1;
            }
        }
    if(test==0)
        for (G4int i=0;i<m_count;i++)  {
            if ( command==pStripWidthY[i] ) {
                GetGridDiscretization()->SetStripWidthY(m_name[i], pStripWidthY[i]->GetNewDoubleValue(newValue));
                test=1;
            }
        }
    if(test==0)
        for (G4int i=0;i<m_count;i++)  {
            if ( command== pNumberStripsX[i] ) {
                GetGridDiscretization()->SetNumberStripsX(m_name[i], pNumberStripsX[i]->GetNewIntValue(newValue));
                test=1;
            }
        }
    if(test==0)
        for (G4int i=0;i<m_count;i++)  {
            if ( command== pNumberStripsY[i] ) {
                GetGridDiscretization()->SetNumberStripsY(m_name[i], pNumberStripsY[i]->GetNewIntValue(newValue));
                test=1;
            }
        }
    if(test==0)
        for (G4int i=0;i<m_count;i++)  {
            if ( command== pNumberReadOutBlocksX[i] ) {
                GetGridDiscretization()->SetNumberReadOutBlocksX(m_name[i], pNumberReadOutBlocksX[i]->GetNewIntValue(newValue));
                test=1;
            }
        }
    if(test==0)
        for (G4int i=0;i<m_count;i++)  {
            if ( command== pNumberReadOutBlocksY[i] ) {
                GetGridDiscretization()->SetNumberReadOutBlocksY(m_name[i], pNumberReadOutBlocksY[i]->GetNewIntValue(newValue));
                test=1;
            }
        }
   // if(test==0)
   //     for (G4int i=0;i<m_count;i++)  {
   //       if ( command==pRejectionMultiplesCmd[i] ) {
   //          GetGridDiscretization()->SetRejectionFlag(m_name[i], pRejectionMultiplesCmd[i]->GetNewBoolValue(newValue));
   //     test=1;
    //      }
    //    }
    if(test==0)
        GatePulseProcessorMessenger::SetNewValue(command,newValue);

  //Estas dos dan buenos valores
//         G4cout<<" mNAme 0"<<m_name[0]<<G4endl;
//        G4cout<<" newValue messenger2 "<< GetGridDiscretization()->m_table[m_name[0]].numberStripsX<<G4endl;




}
