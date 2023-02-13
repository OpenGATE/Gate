/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateOPETSystem.hh"

#include "G4UnitsTable.hh"

#include "GateClockDependentMessenger.hh"
#include "GateBox.hh"
#include "GateWedge.hh"
#include "GateSystemComponent.hh"
#include "GateVolumePlacement.hh"
#include "GateCoincidenceSorter.hh"
#include "GateOutputMgr.hh"

#include "GateConfiguration.h"
#ifdef GATE_USE_LMF
#include "GateToLMF.hh"
#endif
#include "GateDigitizerMgr.hh"

// Constructor
GateOPETSystem::GateOPETSystem(const G4String& itsName)
: GateVSystem( itsName , true )
{
  // Set up a messenger
  m_messenger = new GateClockDependentMessenger(this);
  m_messenger->SetDirectoryGuidance(G4String("Controls the system '") + GetObjectName() + "'" );
  
  // Define the scanner components
  GateBoxComponent* rSectorComponent = new GateBoxComponent("rsector",GetBaseComponent(),this);
  GateArrayComponent* moduleComponent = new GateArrayComponent("module",rSectorComponent,this);
  GateArrayComponent* submoduleComponent = new GateArrayComponent("submodule",moduleComponent,this);
//  GateArrayComponent* boxcrystalComponent = new GateArrayComponent("boxcrystal",submoduleComponent,this);
  GateArrayComponent* wedgecrystalComponent = new GateArrayComponent("wedge",submoduleComponent,this);
    new GateWedgeComponent("layer0",wedgecrystalComponent,this);
    new GateWedgeComponent("layer1",wedgecrystalComponent,this);
    new GateWedgeComponent("layer2",wedgecrystalComponent,this);
    new GateWedgeComponent("layer3",wedgecrystalComponent,this);
    new GateWedgeComponent("layer4",wedgecrystalComponent,this);
    new GateWedgeComponent("layer5",wedgecrystalComponent,this);
    new GateWedgeComponent("layer6",wedgecrystalComponent,this);
    new GateWedgeComponent("layer7",wedgecrystalComponent,this);

 
  // Integrate a coincidence sorter into the digitizer
    //OK GND 2022
    GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();
    GateCoincidenceSorter* coincidenceSorter = new GateCoincidenceSorter(digitizerMgr,"Coincidences");
    digitizerMgr->AddNewCoincidenceSorter(coincidenceSorter);
#ifdef GATE_USE_LMF

  // Insert an LMF output module into the output manager
  GateOutputMgr *outputMgr = GateOutputMgr::GetInstance();
  GateToLMF* gateToLMF1 = new GateToLMF("lmf1", outputMgr,this,GateOutputMgr::GetDigiMode()); // this for geometry
  outputMgr->AddOutputModule((GateVOutputModule*)gateToLMF1);
 
#endif 

  SetOutputIDName((char *)"gantryID",0);
  SetOutputIDName((char *)"rsectorID",1);
  SetOutputIDName((char *)"moduleID",2);
  SetOutputIDName((char *)"submoduleID",3);
  SetOutputIDName((char *)"crystalID",4);
  SetOutputIDName((char *)"layerID",5);

}



// Destructor
GateOPETSystem::~GateOPETSystem() 
{
  delete m_messenger;
}



/*  Method overloading the base-class virtual method Describe().
    This methods prints-out a description of the system, which is
    optimised for creating LMF header files

	indent: the print-out indentation (cosmetic parameter)
*/    
void GateOPETSystem::Describe(size_t indent)
{
  GateVSystem::Describe(indent);
  PrintToStream(G4cout,true);
}



/* Method overloading the base-class virtual method Describe().
   This methods prints out description of the system to a stream.
   It is essentially to be used by the class GateToLMF, but it may also be used by Describe()

	aStream: the output stream
	doPrintNumbers: tells whether we print-out the volume numbers in addition to their dimensions
*/    
void GateOPETSystem::PrintToStream(std::ostream& aStream,G4bool doPrintNumbers)
{
  aStream << "geometrical design type: " << 1   	      	      	      	      	  << Gateendl;

  aStream << "ring diameter: " << G4BestUnit( 2*ComputeInternalRadius() ,"Length")   	      	      	      	      	  << Gateendl;

  GateBoxComponent* rsectorComponent = FindBoxCreatorComponent("rsector");

  G4double rsectorAxialPitch = rsectorComponent->GetLinearRepeatVector().z();
  aStream << "rsector axial pitch: " << G4BestUnit( rsectorAxialPitch ,"Length")  	  << Gateendl;
  
  G4double rsectorAzimuthalPitch = rsectorComponent->GetAngularRepeatPitch();
  aStream << "rsector azimuthal pitch: " << rsectorAzimuthalPitch/degree << " degree"  	 	  << Gateendl;    

  G4double rsectorRadialSize     = rsectorComponent->GetBoxLength(0) ;
  G4double rsectorTangentialSize = rsectorComponent->GetBoxLength(1) ;
  G4double rsectorAxialSize      = rsectorComponent->GetBoxLength(2) ;
  aStream << "rsector tangential size: " << G4BestUnit( rsectorTangentialSize ,"Length")  	  << Gateendl;
  aStream << "rsector axial size: " << G4BestUnit( rsectorAxialSize ,"Length")  	      	  << Gateendl;

  GateArrayComponent* moduleComponent = FindArrayComponent("module");

  GateBox* moduleCreator =   moduleComponent->GetBoxCreator();	      	      	      	      	  
  G4double moduleRadialSize     = moduleCreator ? moduleCreator->GetBoxLength(0) : rsectorRadialSize ;
  G4double moduleTangentialSize = moduleCreator ? moduleCreator->GetBoxLength(1) : rsectorTangentialSize ;
  G4double moduleAxialSize      = moduleCreator ? moduleCreator->GetBoxLength(2) : rsectorAxialSize ;
  aStream << "module axial size: " << G4BestUnit( moduleAxialSize ,"Length")  	      	  << Gateendl;
  aStream << "module tangential size: " << G4BestUnit( moduleTangentialSize ,"Length")  	  << Gateendl;

  G4ThreeVector modulePitchVector = moduleComponent->GetRepeatVector(); 
  aStream << "module axial pitch: " << G4BestUnit( modulePitchVector.z() ,"Length")    	  << Gateendl;
  aStream << "module tangential pitch: " << G4BestUnit( modulePitchVector.y() ,"Length")    	  << Gateendl;

  GateArrayComponent* submoduleComponent = FindArrayComponent("submodule");

  GateBox* submoduleCreator =   submoduleComponent->GetBoxCreator();	      	      	      	      	  
  G4double submoduleRadialSize     = submoduleCreator ? submoduleCreator->GetBoxLength(0) : moduleRadialSize ;
  G4double submoduleTangentialSize = submoduleCreator ? submoduleCreator->GetBoxLength(1) : moduleTangentialSize ;
  G4double submoduleAxialSize      = submoduleCreator ? submoduleCreator->GetBoxLength(2) : moduleAxialSize ;
  aStream << "submodule axial size: " << G4BestUnit( submoduleAxialSize ,"Length")  	      	  << Gateendl;
  aStream << "submodule tangential size: " << G4BestUnit( submoduleTangentialSize ,"Length")  	  << Gateendl;

  G4ThreeVector submodulePitchVector = submoduleComponent->GetRepeatVector(); 
  aStream << "submodule axial pitch: " << G4BestUnit( submodulePitchVector.z() ,"Length")    	  << Gateendl;
  aStream << "submodule tangential pitch: " << G4BestUnit( submodulePitchVector.y() ,"Length")     << Gateendl;

  GateArrayComponent* wedgecrystalComponent   = FindArrayComponent("wedge");

  GateBox* wedgecrystalCreator =   wedgecrystalComponent->GetBoxCreator();	      	      	      	      	  
  G4double crystalRadialSize     = wedgecrystalCreator ? wedgecrystalCreator->GetBoxLength(0) : submoduleRadialSize ;
  G4double crystalTangentialSize = wedgecrystalCreator ? wedgecrystalCreator->GetBoxLength(1) : submoduleTangentialSize ;
  G4double crystalAxialSize      = wedgecrystalCreator ? wedgecrystalCreator->GetBoxLength(2) : submoduleAxialSize ;
  aStream << "crystal radial size: " << G4BestUnit( crystalRadialSize ,"Length")  	      	  << Gateendl;
  aStream << "crystal axial size: " << G4BestUnit( crystalAxialSize ,"Length")  	      	  << Gateendl;
  aStream << "crystal tangential size: " << G4BestUnit( crystalTangentialSize ,"Length")  	  << Gateendl;

  G4ThreeVector crystalPitchVector = wedgecrystalComponent->GetRepeatVector() ; 
  aStream << "crystal axial pitch: " << G4BestUnit( crystalPitchVector.z() ,"Length")    	  << Gateendl;
  aStream << "crystal tangential pitch: " << G4BestUnit( crystalPitchVector.y() ,"Length")    	  << Gateendl;

  size_t NbLayers = wedgecrystalComponent->GetActiveChildNumber();
  //for (size_t i=0; i<NbLayers; i++) {

    //char buffer[80];
    //sprintf(buffer,"layer%u",i);
    //GateBoxCreatorComponent* layerComponent = FindBoxCreatorComponent(buffer);
    //aStream << buffer << " radial size: " << G4BestUnit( layerComponent->GetBoxLength(0) ,"Length") << Gateendl;
    //aStream << "in " << buffer << " interaction length: " << G4BestUnit( 0.5*layerComponent->GetBoxLength(0) ,"Length") << Gateendl;
  //}



  if (doPrintNumbers) {
    aStream << "Axial nb of rsectors: " << rsectorComponent->GetLinearRepeatNumber()     	      	      	      	      	      	      	      	  << Gateendl;
    aStream << "Azimuthal nb of rsectors: " << rsectorComponent->GetAngularRepeatNumber()   	      	      	      	      	      	      	      	  << Gateendl;
    aStream << "Axial nb of modules: " << moduleComponent->GetRepeatNumber(2) 	      	      	      	  	  << Gateendl;
    aStream << "Tangential nb of modules: " << moduleComponent->GetRepeatNumber(1) 	      	      	      	  << Gateendl;
    aStream << "Axial nb of submodules: " << submoduleComponent->GetRepeatNumber(2) 	      	      	      	  << Gateendl;
    aStream << "Tangential nb of submodules: " << submoduleComponent->GetRepeatNumber(1) 	      	      	  	  << Gateendl;
    aStream << "Axial nb of crystals: " << wedgecrystalComponent->GetRepeatNumber(2) 	      	      	      	  	  << Gateendl;
    aStream << "Tangential nb of crystals: " << wedgecrystalComponent->GetRepeatNumber(1) 	      	      	      	  << Gateendl;
    aStream << "Radial nb of layers: " << NbLayers 	      	      	      	      	      	      	      	      	      	      	  << Gateendl;
  }
}


// Compute the internal radius of the crystal ring.
G4double GateOPETSystem::ComputeInternalRadius()
{
  // Compute the radius to the center of the rsector
  GateBoxComponent *rsector = FindBoxCreatorComponent("rsector");
  GateVolumePlacement *rsectorMove = rsector->FindPlacementMove();
  G4double radius = rsectorMove ? rsectorMove->GetTranslation().x() : 0.;

  // Decrease by the rsector half-length to get the internal radius
  radius -=  .5 * rsector->GetBoxLength(0);

  // Add all the offsets between innermost edges
  radius += FindComponent("module")->ComputeOffset(0,GateSystemComponent::align_left,GateSystemComponent::align_left);
  radius += FindComponent("submodule")->ComputeOffset(0,GateSystemComponent::align_left,GateSystemComponent::align_left);
  radius += FindComponent("wedge")->ComputeOffset(0,GateSystemComponent::align_left,GateSystemComponent::align_left);

  // Add the offset to the innermost edge of the first layer
  radius += FindComponent("layer0")->ComputeOffset(0,GateSystemComponent::align_left,GateSystemComponent::align_left);
    
  return radius;
}



