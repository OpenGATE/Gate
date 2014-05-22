/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVoxelBoxParam.hh"
#include "GateGeometryVoxelImageReader.hh"
#include "GateVoxelBoxParameterization.hh"
#include "GateVoxelBoxParameterized.hh"
#include "G4PVParameterised.hh"
#include "GateBox.hh"

//---------------------------------------------------------------------------------------------------------------------
// Constructor
GateVoxelBoxParam::GateVoxelBoxParam(const  G4String& itsName, GateVoxelBoxParameterized* vpi)
  : GateBox(itsName,"Vacuum",1,1,1,false,false),
    itsInserter(vpi), m_parameterization(0), m_pvParameterized(0)
{
}



// Destructor
GateVoxelBoxParam::~GateVoxelBoxParam()
{
}

//
void GateVoxelBoxParam::ConstructOwnPhysicalVolume(G4bool flagUpdate){
  // G4cout << "GateVoxelBoxParam::ConstructOwnPhysicalVolumes - Entered, name "<< mName <<", flag "<< std::boolalpha << flagUpdate <<  G4endl<<std::flush;
  
  // For the update case; there is nothing to do here.
  if (flagUpdate) return;
    
  DestroyGeometry();  
  
  // Build the parameterization
  GateVGeometryVoxelReader* itsReader ( itsInserter->GetReader() );
  G4ThreeVector voxelSize(  itsReader->GetVoxelSize()  );
  G4ThreeVector voxelNumber(  itsReader->GetVoxelNx(),itsReader->GetVoxelNy(),itsReader->GetVoxelNz()  );
  m_parameterization = new GateVoxelBoxParameterization(itsReader, voxelNumber, voxelSize );

  //  Suggestion by J. Apostolakis to reduce memory consumption
  itsInserter->GetCreator()->GetLogicalVolume()->SetSmartless(0.02);

  // Build the physical volume
  m_pvParameterized = new G4PVParameterised(mName+"_PVP",
					    GetCreator()->GetLogicalVolume(),
					    itsInserter->GetCreator()->GetLogicalVolume(),
					    kUndefined,
					    m_parameterization->GetNbOfCopies(),
					    m_parameterization
					    );
  PushPhysicalVolume(m_pvParameterized);
  
}


void GateVoxelBoxParam::DestroyGeometry(){
  
  // G4cout << "GateVoxelBoxParam::DestructOwnPhysicalVolumes - Entered"<<G4endl<<std::flush;
  if (m_parameterization) {
    delete m_parameterization;
  }
  m_parameterization=0;
  
  m_pvParameterized=0;
  GateVVolume::DestroyGeometry();
}
