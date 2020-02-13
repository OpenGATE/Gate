/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateDoIModelsMessenger.hh"

#include "GateDoIModels.hh"

#include "G4UIcmdWith3Vector.hh"
#include "GateDualLayerLaw.hh"
#include "GateDoIBlurrNegExpLaw.hh"

GateDoIModelsMessenger::GateDoIModelsMessenger(GateDoIModels* itsDoIModel)
    : GatePulseProcessorMessenger(itsDoIModel)
{
    G4String guidance;
    G4String cmdName;
    G4String cmdName2;

    cmdName = GetDirectoryName() + "setAxis";
    axisCmd = new G4UIcmdWith3Vector(cmdName,this);
    axisCmd->SetGuidance("Set the DoI direction. Selecting one of the axis (only possible options X, Y or Z directions)");

    cmdName2 = GetDirectoryName() + "setDoIModel";
    lawCmd = new G4UIcmdWithAString(cmdName2,this);
    lawCmd->SetGuidance("Set the DoI model ");
}







GateDoIModelsMessenger::~GateDoIModelsMessenger()
{
    delete axisCmd;
    delete lawCmd;
}


GateVDoILaw* GateDoIModelsMessenger::CreateDoILaw(const G4String& law) {

    if ( law == "dualLayer" ) {
        return new GateDualLayerLaw(GetDoIModel()->GetObjectName()+ G4String("/dualLayer"));

    } else if ( law == "DoIBlurrNegExp" ) {
       return new GateDoIBlurrNegExpLaw(GetDoIModel()->GetObjectName() + G4String("/DoIBlurrNegExp"));
    } else {
        G4cerr << "No match for '" << law << "'DoI law.\n";
        G4cerr << "Candidates are: dual layer, ..\n";
    }

    return NULL;
}

void GateDoIModelsMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==axisCmd )
    { GetDoIModel()->SetDoIAxis(axisCmd->GetNew3VectorValue(newValue)); }
  else if (command==lawCmd ){
      GateVDoILaw* a_DoILaw = CreateDoILaw(newValue);
              if (a_DoILaw != NULL) {
                  GetDoIModel()->SetDoILaw(a_DoILaw);
              }
  }

  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
