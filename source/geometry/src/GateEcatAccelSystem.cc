/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GateEcatAccelSystem.hh"
#include "GateClockDependentMessenger.hh"
#include "GateBox.hh"
#include "GateSystemComponent.hh"
#include "GateVolumePlacement.hh"
#include "GateCoincidenceSorter.hh"
#include "GateToSinoAccel.hh"
#include "GateSinoAccelToEcat7.hh"
#include "GateDigitizerMgr.hh"
#include "GateOutputMgr.hh"

#include "G4UnitsTable.hh"

// Constructor
GateEcatAccelSystem::GateEcatAccelSystem(const G4String& itsName)
  : GateVSystem( itsName , true ),
    m_gateToSinoAccel(0)
{
  // Set up a messenger
  m_messenger = new GateClockDependentMessenger(this);
  m_messenger->SetDirectoryGuidance(G4String("Controls the system '") + GetObjectName() + "'" );

  // Define the scanner components
  GateBoxComponent* blockComponent   = new GateBoxComponent("block",GetBaseComponent(),this);
  /*GateArrayComponent* arrayComponent =  */ new GateArrayComponent("crystal",blockComponent,this);

  // Integrate a coincidence sorter into the digitizer
  //OK GND 2022
  GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();
  GateCoincidenceSorter* coincidenceSorter = new GateCoincidenceSorter(digitizerMgr,"Coincidences");
  digitizerMgr->AddNewCoincidenceSorter(coincidenceSorter);

  // Insert a sinogram maker and a ECAT7 writer into the output manager
  GateOutputMgr *outputMgr = GateOutputMgr::GetInstance();
  m_gateToSinoAccel = new GateToSinoAccel("sinoAccel", outputMgr,this,GateOutputMgr::GetDigiMode());
  outputMgr->AddOutputModule((GateVOutputModule*)m_gateToSinoAccel);
#ifdef GATE_USE_ECAT7
  m_gateSinoAccelToEcat7 = new GateSinoAccelToEcat7("ecat7", outputMgr,this,GateOutputMgr::GetDigiMode());
  outputMgr->AddOutputModule((GateVOutputModule*)m_gateSinoAccelToEcat7);
#endif

  SetOutputIDName((char *)"gantryID",0);
  SetOutputIDName((char *)"blockID",1);
  SetOutputIDName((char *)"crystalID",2);

}



// Destructor
GateEcatAccelSystem::~GateEcatAccelSystem()
{
  delete m_messenger;
}



/*  Method overloading the base-class virtual method Describe().
    This methods prints-out a description of the system, which is
    optimised for creating ECAT header files

    indent: the print-out indentation (cosmetic parameter)
*/
void GateEcatAccelSystem::Describe(size_t indent)
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
void GateEcatAccelSystem::PrintToStream(std::ostream& aStream,G4bool doPrintNumbers)
{
  aStream << " >> geometrical design type: " << "ECAT ACCEL\n";

  aStream << " >> ring diameter: " << G4BestUnit( 2*ComputeInternalRadius() ,"Length")   << Gateendl;

  GateBoxComponent* blockComponent = FindBoxCreatorComponent("block");

  G4double blockAxialPitch = blockComponent->GetSphereAxialRepeatPitch();
  aStream << " >> block axial pitch: " << G4BestUnit( blockAxialPitch ,"Length")  	  << Gateendl;

  G4double blockAzimuthalPitch = blockComponent->GetSphereAzimuthalRepeatPitch();
  aStream << " >> block azimuthal pitch: " << blockAzimuthalPitch/degree << " degree"  	 	  << Gateendl;

  G4double blockTangentialSize = blockComponent->GetBoxLength(0) ;
  G4double blockAxialSize      = blockComponent->GetBoxLength(2) ;
  aStream << " >> block tangential size: " << G4BestUnit( blockTangentialSize ,"Length")  	  << Gateendl;
  aStream << " >> block axial size: " << G4BestUnit( blockAxialSize ,"Length")  	      	  << Gateendl;

  GateArrayComponent* crystalComponent = FindArrayComponent("crystal");

  G4double crystalTangentialSize = crystalComponent->GetBoxLength(0);
  G4double crystalAxialSize      = crystalComponent->GetBoxLength(2);
  aStream << " >> crystal axial size: " << G4BestUnit( crystalAxialSize ,"Length")  	      	  << Gateendl;
  aStream << " >> crystal tangential size: " << G4BestUnit( crystalTangentialSize ,"Length")  	  << Gateendl;

  G4ThreeVector crystalPitchVector = crystalComponent->GetRepeatVector();
  aStream << " >> crystal axial pitch: " << G4BestUnit( crystalPitchVector.z() ,"Length")    	  << Gateendl;
  aStream << " >> crystal tangential pitch: " << G4BestUnit( crystalPitchVector.x() ,"Length")    	  << Gateendl;


  if (doPrintNumbers) {
    aStream << " >> axial nb of blocks: " << blockComponent->GetSphereAxialRepeatNumber()   	   << Gateendl;
    aStream << " >> azimuthal nb of blocks: " << blockComponent->GetSphereAzimuthalRepeatNumber()  << Gateendl;
    aStream << " >> axial nb of crystals: " << crystalComponent->GetRepeatNumber(2) 	      	   << Gateendl;
    aStream << " >> tangential nb of crystals: " << crystalComponent->GetRepeatNumber(0) 	   << Gateendl;
  }
}


// Compute the internal radius of the crystal ring.
G4double GateEcatAccelSystem::ComputeInternalRadius()
{
  // Compute the radius to the center of the block
  GateBoxComponent *block = FindBoxCreatorComponent("block");
  G4double sphereRadius = block->GetSphereRadius();
  sphereRadius -= .5*block->GetBoxLength(1);

  return sphereRadius;
}
