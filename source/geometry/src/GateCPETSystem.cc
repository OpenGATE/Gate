/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateCPETSystem.hh"

#include "G4UnitsTable.hh"

#include "GateClockDependentMessenger.hh"
#include "GateBox.hh"
#include "GateSystemComponent.hh"
#include "GateVolumePlacement.hh"
#include "GateDigitizerMgr.hh"
#include "GateCoincidenceSorter.hh"

// Constructor
GateCPETSystem::GateCPETSystem(const G4String& itsName)
: GateVSystem( itsName , true )
{
  // Changer la profondeur pour le secteur:
  // m_mainComponentDepth = ...;
  
  // Set up a messenger
  m_messenger = new GateClockDependentMessenger(this);
  m_messenger->SetDirectoryGuidance(G4String("Controls the system '") + GetObjectName() + "'" );
  
  // Define the scanner components
  GateCylinderComponent* rSectorComponent =  new GateCylinderComponent("sector",GetBaseComponent(),this);
  GateCylinderComponent* rCassetteComponent =  new GateCylinderComponent("cassette",rSectorComponent,this);
  GateArrayComponent* moduleComponent = new GateArrayComponent("module",rCassetteComponent,this);
  GateArrayComponent* submoduleComponent = new GateArrayComponent("crystal",moduleComponent,this);
  new GateArrayComponent("layer0",submoduleComponent,this);
  new GateArrayComponent("layer1",submoduleComponent,this);
  new GateArrayComponent("layer2",submoduleComponent,this);
  new GateArrayComponent("layer3",submoduleComponent,this);

  // Integrate a coincidence sorter into the digitizer
  //OK GND 2022
  GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();
  GateCoincidenceSorter* coincidenceSorter = new GateCoincidenceSorter(digitizerMgr,"Coincidences");
  digitizerMgr->AddNewCoincidenceSorter(coincidenceSorter);
  
  SetOutputIDName((char *)"gantryID",0);
  SetOutputIDName((char *)"sectorID",1);
  SetOutputIDName((char *)"cassetteID",2);
  SetOutputIDName((char *)"moduleID",3);
  SetOutputIDName((char *)"crystalID",4);
  SetOutputIDName((char *)"layerID",5);

}



// Destructor
GateCPETSystem::~GateCPETSystem() 
{
  delete m_messenger;
}



/*  Method overloading the base-class virtual method Describe().
    This methods prints-out a description of the system, which is
    optimised for creating LMF header files

	indent: the print-out indentation (cosmetic parameter)
*/    
void GateCPETSystem::Describe(size_t indent)
{
  GateVSystem::Describe(indent);
  //PrintToStream(G4cout,true);
}



/* Method overloading the base-class virtual method Describe().
   This methods prints out description of the system to a stream.
   It is essentially to be used by the class GateToLMF, but it may also be used by Describe()

	aStream: the output stream
	doPrintNumbers: tells whether we print-out the volume numbers in addition to their dimensions
*/    
void GateCPETSystem::PrintToStream(std::ostream& aStream,G4bool)
{
  aStream << "geometrical design type: " << "CPET"     	      	      	      	  << Gateendl;

  GateCylinderComponent* crystalComponent = FindCylinderCreatorComponent("crystal");
  G4double crystalHeight = crystalComponent->GetCylinderHeight();
  aStream << "crystal height: " << G4BestUnit( crystalHeight ,"Length")  	  << Gateendl;
  G4double crystalRmin = crystalComponent->GetCylinderRmin();
  aStream << "crystal radius min: " << G4BestUnit( crystalRmin ,"Length")  	  << Gateendl;
  G4double crystalRmax = crystalComponent->GetCylinderRmax();
  aStream << "crystal radius max: " << G4BestUnit( crystalRmax ,"Length")  	  << Gateendl;
  G4double crystalSPhi = crystalComponent->GetCylinderSPhi();
  aStream << "crystal start angle: " << crystalSPhi / degree << " deg\n";
  G4double crystalDPhi = crystalComponent->GetCylinderDPhi();
  aStream << "crystal angular span: " << crystalDPhi / degree << " deg\n";
}


// Compute the internal radius of the crystal ring.
G4double GateCPETSystem::ComputeInternalRadius()
{
  return FindCylinderCreatorComponent("crystal")->GetCylinderRmin();
}




