/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
/*! \file GateImageDeformActorMessenger.cc
    \brief Implementation of GateImageDeformActorMessenger
    \author yannick.lemarechal@univ-brest.fr
	    david.sarrut@creatis.insa-lyon.fr
*/

#include "GateImageDeformActorMessenger.hh"
#include "GateImageDeformActor.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcommand.hh"
//--------------------------------------------------------------------------------------------
GateImageDeformActorMessenger::GateImageDeformActorMessenger(GateImageDeformActor * v)
: GateActorMessenger(v) {
  
    mDeform = v;
    BuildCommands(baseName+v->GetObjectName());
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
GateImageDeformActorMessenger::~GateImageDeformActorMessenger() {
    delete pName;
    delete pSetPDFFile;
    delete mInitialization;
}
//--------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateImageDeformActorMessenger::BuildCommands(G4String base)
{
    G4String  n ;
    G4String guid;
    n = base+"/setPDFFile";
    pSetPDFFile = new G4UIcmdWithAString(n, this);
    guid = G4String("Name of PDF file containing the list of timestamps and associated CT files");
    pSetPDFFile->SetGuidance(guid);
    G4cout<<"\033[32;01m"<<n<<"\033[00m\n"<<G4endl;
}

//--------------------------------------------------------------------------------------------
void GateImageDeformActorMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
    if( command == pSetPDFFile ) 
    {
        mDeform->SetFilename(newValue);
    }

    GateActorMessenger::SetNewValue(command,newValue);
}
//--------------------------------------------------------------------------------------------
