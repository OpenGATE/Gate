/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*! \file GateOpticalSystem.cc
   Created on   2012/07/09  by vesna.cuplov@gmail.com
   Implemented new class GateOpticalSystem for Optical photons: very similar to SPECT. 
    - 3 components: the base, the crystal and the pixel
    - The base is the OPTICAL camera head itself. 
    - The level below is the crystal, which can be monoblock or pixellated.
    - The level below is optional, and is meant to be used for pixellated cameras.
*/  


#include "GateOpticalSystem.hh"
#include "G4UnitsTable.hh"
#include "GateClockDependentMessenger.hh"
#include "GateOutputMgr.hh"
#include "GateToProjectionSet.hh"
#include "GateToOpticalRaw.hh"


// Constructor
GateOpticalSystem::GateOpticalSystem(const G4String& itsName)
: GateVSystem( itsName , false ),
   m_gateToProjectionSet(0),
  m_gateToOpticalRaw(0)
{
  
  // Setup a messenger
  m_messenger = new GateClockDependentMessenger(this);
  m_messenger->SetDirectoryGuidance(G4String("Controls the system '") + GetObjectName() + "'" );
	
  // Define the scanner components
  m_crystalComponent = new GateSystemComponent("crystal",GetBaseComponent(),this);
  m_pixelComponent = new GateSystemComponent("pixel",m_crystalComponent,this);

  // Insert a projection-set maker and a Interfile writer into the output manager
  GateOutputMgr *outputMgr = GateOutputMgr::GetInstance();
  m_gateToProjectionSet = new GateToProjectionSet("projection", outputMgr,this,GateOutputMgr::GetDigiMode()); 
  outputMgr->AddOutputModule((GateVOutputModule*)m_gateToProjectionSet);
  m_gateToOpticalRaw = new GateToOpticalRaw("opticalraw", outputMgr,this,GateOutputMgr::GetDigiMode()); 
  outputMgr->AddOutputModule((GateVOutputModule*)m_gateToOpticalRaw);

  SetOutputIDName((char *)"headID",0);
  SetOutputIDName((char *)"crystalID",1);
  SetOutputIDName((char *)"pixelID",2);
}


// Destructor
GateOpticalSystem::~GateOpticalSystem() 
{
  delete m_messenger;
 
}
