/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCylindricalPETSystem.hh"

#include "G4UnitsTable.hh"

#include "GateClockDependentMessenger.hh"
#include "GateBox.hh"
#include "GateSystemComponent.hh"
#include "GateVolumePlacement.hh"
#include "GateCoincidenceSorter.hh"
#include "GateOutputMgr.hh"

#include "GateConfiguration.h"

#ifdef GATE_USE_LMF
#include "GateToLMF.hh"
#endif

#include "GateDigitizer.hh"

#include "GateCylindricalPETSystemMessenger.hh"

// Constructor
GateCylindricalPETSystem::GateCylindricalPETSystem(const G4String& itsName)
: GateVSystem( itsName , true )
{
  // Set up a messenger
  m_messenger = new GateClockDependentMessenger(this);
  m_messenger->SetDirectoryGuidance(G4String("Controls the system '") + GetObjectName() + "'" );
  
	// Set up a messenger
  m_messenger2 = new GateCylindricalPETSystemMessenger(this);
  m_messenger2->SetDirectoryGuidance(G4String("Controls the system '") + GetObjectName() + "'" );

  // Define the scanner components
  GateBoxComponent* rSectorComponent = new GateBoxComponent("rsector",GetBaseComponent(),this);
  GateArrayComponent* moduleComponent = new GateArrayComponent("module",rSectorComponent,this);
  GateArrayComponent* submoduleComponent = new GateArrayComponent("submodule",moduleComponent,this);
  GateArrayComponent* crystalComponent = new GateArrayComponent("crystal",submoduleComponent,this);
  new GateBoxComponent("layer0",crystalComponent,this);
  new GateBoxComponent("layer1",crystalComponent,this);
  new GateBoxComponent("layer2",crystalComponent,this);
  new GateBoxComponent("layer3",crystalComponent,this);

  // Integrate a coincidence sorter into the digitizer
  G4double coincidenceWindow = 10.* ns;
  GateDigitizer* digitizer = GateDigitizer::GetInstance();
  GateCoincidenceSorter* coincidenceSorter = new GateCoincidenceSorter(digitizer,"Coincidences",coincidenceWindow);
  digitizer->StoreNewCoincidenceSorter(coincidenceSorter);
  
#ifdef GATE_USE_LMF

  // Insert an LMF output module into the output manager
  GateOutputMgr *outputMgr = GateOutputMgr::GetInstance();
  GateToLMF* gateToLMF1 = new GateToLMF("lmf", outputMgr,this,GateOutputMgr::GetDigiMode()); // this for geometry
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
GateCylindricalPETSystem::~GateCylindricalPETSystem() 
{
  delete m_messenger;
}



/*  Method overloading the base-class virtual method Describe().
    This methods prints-out a description of the system, which is
    optimised for creating LMF header files

	indent: the print-out indentation (cosmetic parameter)
*/    
void GateCylindricalPETSystem::Describe(size_t indent)
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
void GateCylindricalPETSystem::PrintToStream(std::ostream& aStream,G4bool doPrintNumbers)
{
  aStream << "geometrical design type: " << 1   	      	      	      	      	  << G4endl;

  aStream << "ring diameter: " << G4BestUnit( 2*ComputeInternalRadius() ,"Length")   	      	      	      	      	  << G4endl;

  GateBoxComponent* rsectorComponent = FindBoxCreatorComponent("rsector");

  G4double rsectorAxialPitch = rsectorComponent->GetLinearRepeatVector().z();
  aStream << "rsector axial pitch: " << G4BestUnit( rsectorAxialPitch ,"Length")  	  << G4endl;
  
  G4double rsectorAzimuthalPitch = rsectorComponent->GetAngularRepeatPitch();
  aStream << "rsector azimuthal pitch: " << rsectorAzimuthalPitch/degree << " degree"  	 	  << G4endl;    

  G4double rsectorRadialSize     = rsectorComponent->GetBoxLength(0) ;
  G4double rsectorTangentialSize = rsectorComponent->GetBoxLength(1) ;
  G4double rsectorAxialSize      = rsectorComponent->GetBoxLength(2) ;
  aStream << "rsector tangential size: " << G4BestUnit( rsectorTangentialSize ,"Length")  	  << G4endl;
  aStream << "rsector axial size: " << G4BestUnit( rsectorAxialSize ,"Length")  	      	  << G4endl;

  GateArrayComponent* moduleComponent = FindArrayComponent("module");

  GateBox* moduleCreator =   moduleComponent->GetBoxCreator();	      	      	      	      	  
  G4double moduleRadialSize     = moduleCreator ? moduleCreator->GetBoxLength(0) : rsectorRadialSize ;
  G4double moduleTangentialSize = moduleCreator ? moduleCreator->GetBoxLength(1) : rsectorTangentialSize ;
  G4double moduleAxialSize      = moduleCreator ? moduleCreator->GetBoxLength(2) : rsectorAxialSize ;
  aStream << "module axial size: " << G4BestUnit( moduleAxialSize ,"Length")  	      	  << G4endl;
  aStream << "module tangential size: " << G4BestUnit( moduleTangentialSize ,"Length")  	  << G4endl;

  G4ThreeVector modulePitchVector = moduleComponent->GetRepeatVector(); 
  aStream << "module axial pitch: " << G4BestUnit( modulePitchVector.z() ,"Length")    	  << G4endl;
  aStream << "module tangential pitch: " << G4BestUnit( modulePitchVector.y() ,"Length")    	  << G4endl;

  GateArrayComponent* submoduleComponent = FindArrayComponent("submodule");

  GateBox* submoduleCreator =   submoduleComponent->GetBoxCreator();	      	      	      	      	  
  G4double submoduleRadialSize     = submoduleCreator ? submoduleCreator->GetBoxLength(0) : moduleRadialSize ;
  G4double submoduleTangentialSize = submoduleCreator ? submoduleCreator->GetBoxLength(1) : moduleTangentialSize ;
  G4double submoduleAxialSize      = submoduleCreator ? submoduleCreator->GetBoxLength(2) : moduleAxialSize ;
  aStream << "submodule axial size: " << G4BestUnit( submoduleAxialSize ,"Length")  	      	  << G4endl;
  aStream << "submodule tangential size: " << G4BestUnit( submoduleTangentialSize ,"Length")  	  << G4endl;

  G4ThreeVector submodulePitchVector = submoduleComponent->GetRepeatVector(); 
  aStream << "submodule axial pitch: " << G4BestUnit( submodulePitchVector.z() ,"Length")    	  << G4endl;
  aStream << "submodule tangential pitch: " << G4BestUnit( submodulePitchVector.y() ,"Length")     << G4endl;

  GateArrayComponent* crystalComponent   = FindArrayComponent("crystal");

  GateBox* crystalCreator =   crystalComponent->GetBoxCreator();	      	      	      	      	  
  G4double crystalRadialSize     = crystalCreator ? crystalCreator->GetBoxLength(0) : submoduleRadialSize ;
  G4double crystalTangentialSize = crystalCreator ? crystalCreator->GetBoxLength(1) : submoduleTangentialSize ;
  G4double crystalAxialSize      = crystalCreator ? crystalCreator->GetBoxLength(2) : submoduleAxialSize ;
  aStream << "crystal radial size: " << G4BestUnit( crystalRadialSize ,"Length")  	      	  << G4endl;
  aStream << "crystal axial size: " << G4BestUnit( crystalAxialSize ,"Length")  	      	  << G4endl;
  aStream << "crystal tangential size: " << G4BestUnit( crystalTangentialSize ,"Length")  	  << G4endl;

  G4ThreeVector crystalPitchVector = crystalComponent->GetRepeatVector() ; 
  aStream << "crystal axial pitch: " << G4BestUnit( crystalPitchVector.z() ,"Length")    	  << G4endl;
  aStream << "crystal tangential pitch: " << G4BestUnit( crystalPitchVector.y() ,"Length")    	  << G4endl;

  size_t NbLayers = crystalComponent->GetActiveChildNumber();
  for (size_t i=0; i<NbLayers; i++) {

    char buffer[80];
    sprintf(buffer,"layer%u",(unsigned int)i);
    GateBoxComponent* layerComponent = FindBoxCreatorComponent(buffer);
    aStream << buffer << " radial size: " << G4BestUnit( layerComponent->GetBoxLength(0) ,"Length") << G4endl;
    aStream << "in " << buffer << " interaction length: " << G4BestUnit( 0.5*layerComponent->GetBoxLength(0) ,"Length") << G4endl;
  }



  if (doPrintNumbers) {
    aStream << "Axial nb of rsectors: " << rsectorComponent->GetLinearRepeatNumber()     	      	      	      	      	      	      	      	  << G4endl;
    aStream << "Azimuthal nb of rsectors: " << rsectorComponent->GetAngularRepeatNumber()   	      	      	      	      	      	      	      	  << G4endl;
    aStream << "Axial nb of modules: " << moduleComponent->GetRepeatNumber(2) 	      	      	      	  	  << G4endl;
    aStream << "Tangential nb of modules: " << moduleComponent->GetRepeatNumber(1) 	      	      	      	  << G4endl;
    aStream << "Axial nb of submodules: " << submoduleComponent->GetRepeatNumber(2) 	      	      	      	  << G4endl;
    aStream << "Tangential nb of submodules: " << submoduleComponent->GetRepeatNumber(1) 	      	      	  	  << G4endl;
    aStream << "Axial nb of crystals: " << crystalComponent->GetRepeatNumber(2) 	      	      	      	  	  << G4endl;
    aStream << "Tangential nb of crystals: " << crystalComponent->GetRepeatNumber(1) 	      	      	      	  << G4endl;
    aStream << "Radial nb of layers: " << NbLayers 	      	      	      	      	      	      	      	      	      	      	  << G4endl;
  }
}


// Compute the internal radius of the crystal ring.
G4double GateCylindricalPETSystem::ComputeInternalRadius()
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
  radius += FindComponent("crystal")->ComputeOffset(0,GateSystemComponent::align_left,GateSystemComponent::align_left);

  // Add the offset to the innermost edge of the first layer
  radius += FindComponent("layer0")->ComputeOffset(0,GateSystemComponent::align_left,GateSystemComponent::align_left);
    
  return radius;
}


void GateCylindricalPETSystem::AddNewRSECTOR( G4String aName )
{ m_rsectorID.insert( make_pair(  aName, ++m_maxrsectorID  ) );
 G4String modulename = aName+"module";
 G4String submodulename = aName+"submodule";
 G4String crystalname = aName+"crystal";
 G4String layername = aName+"layer";
  GateBoxComponent* rsectorComponent = new GateBoxComponent(aName , GetBaseComponent(),this);
  GateArrayComponent* moduleComponent = new GateArrayComponent(modulename,rsectorComponent,this);
  GateArrayComponent* submoduleComponent = new GateArrayComponent(submodulename,moduleComponent,this);
  GateArrayComponent* crystalComponent = new GateArrayComponent(crystalname,submoduleComponent,this);
  new GateBoxComponent( layername+"0",crystalComponent,this);
  new GateBoxComponent( layername+"1",crystalComponent,this);
  new GateBoxComponent( layername+"2",crystalComponent,this);
  new GateBoxComponent( layername+"3",crystalComponent,this);
  char nameID[30];
  memset( nameID , '\0', 30 );
  G4String s = aName+"ID";
  s.copy( nameID , s.length() );
G4cout << " m_maxindex at start " << m_maxindex <<G4endl;
G4cout << nameID <<G4endl;
}
