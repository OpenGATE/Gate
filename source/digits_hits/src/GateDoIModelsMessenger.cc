/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateDoIModels.cc for more details

  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com

*/

#include "GateDoIModelsMessenger.hh"
#include "GateDoIModels.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"
#include "GateDualLayerLaw.hh"
#include "GateDoIBlurrNegExpLaw.hh"

#include "G4UIcmdWith3Vector.hh"

GateDoIModelsMessenger::GateDoIModelsMessenger (GateDoIModels* DoIModels)
:GateClockDependentMessenger(DoIModels),
 	 m_DoIModels(DoIModels)
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
	delete  DoICmd;
	delete axisCmd;
	delete lawCmd;
}

GateVDoILaw* GateDoIModelsMessenger::CreateDoILaw(const G4String& law) {

    if ( law == "dualLayer" ) {
        return new GateDualLayerLaw(m_DoIModels->GetObjectName()+ G4String("/dualLayer"));

    } else if ( law == "DoIBlurrNegExp" )
    {
       return new GateDoIBlurrNegExpLaw(m_DoIModels->GetObjectName() + G4String("/DoIBlurrNegExp"));
    } else
    {
    	GateError("\n No match for '" << law << "' DoI law.\n Candidates are: dualLayer, DoIBlurrNegExp, ..");
    }

    return NULL;
}

void GateDoIModelsMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{

	if (aCommand==axisCmd)
	{
	   m_DoIModels->SetDoIAxis(axisCmd->GetNew3VectorValue(newValue));
	}
	else if (aCommand==lawCmd )
	{
	   GateVDoILaw* a_DoILaw = CreateDoILaw(newValue);
	   if (a_DoILaw != NULL)
	   {
	      m_DoIModels->SetDoILaw(a_DoILaw);
	   }
	}
	else
	{
	    GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	}
}













