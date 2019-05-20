/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateScannerSystem.hh"

#include "G4UnitsTable.hh"

#include "GateClockDependentMessenger.hh"

// Constructor
GateScannerSystem::GateScannerSystem(const G4String& itsName)
: GateVSystem( itsName , false )
{

  // Setup a messenger
  m_messenger = new GateClockDependentMessenger(this);
  m_messenger->SetDirectoryGuidance(G4String("Controls the system '") + GetObjectName() + "'" );
  
  // Define the scanner components
  GateSystemComponent* aComponent;
  aComponent = new GateSystemComponent("level1",GetBaseComponent(),this);
  aComponent = new GateSystemComponent("level2",aComponent,this);
  aComponent = new GateSystemComponent("level3",aComponent,this);
  aComponent = new GateSystemComponent("level4",aComponent,this);
  aComponent = new GateSystemComponent("level5",aComponent,this);
  new GateSystemComponent("layer0",aComponent,this);
  new GateSystemComponent("layer1",aComponent,this);
  SetOutputIDName((char *)"level1ID",1);
  SetOutputIDName((char *)"level2ID",2);
  SetOutputIDName((char *)"level3ID",3);
  SetOutputIDName((char *)"level4ID",4);
  SetOutputIDName((char *)"level5ID",5);
  //SetOutputIDName((char *)"layerID",5);
}


// Destructor
GateScannerSystem::~GateScannerSystem() 
{
  delete m_messenger;
}

