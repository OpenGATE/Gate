/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateCompressedVoxelParam.hh"
#include "GateBox.hh"

#include "GateGeometryVoxelImageReader.hh"
#include "GateCompressedVoxelParameterization.hh"
#include "GateCompressedVoxelParameterized.hh"
#include "G4PVParameterised.hh"


// Constructor
GateCompressedVoxelParam::GateCompressedVoxelParam(const  G4String& name, GateCompressedVoxelParameterized* vpi):GateBox(name,"Vacuum",1,1,1,false,false)
,itsInserter(vpi),m_parameterization(0),m_pvParameterized(0)
{ 
  
}


// Destructor
GateCompressedVoxelParam::~GateCompressedVoxelParam()
{
}


//
void GateCompressedVoxelParam::ConstructOwnPhysicalVolume(G4bool flagUpdate){


  // For the update case; there is nothing to do here.
  if (flagUpdate) return;
    
  // Build the parameterization
  GateVGeometryVoxelReader* itsReader ( itsInserter->GetReader() );
  G4ThreeVector voxelSize(  itsReader->GetVoxelSize()  );
  G4ThreeVector voxelNumber(  itsReader->GetVoxelNx(),itsReader->GetVoxelNy(),itsReader->GetVoxelNz()  );
 
 
  m_parameterization = new GateCompressedVoxelParameterization(itsReader, voxelNumber, voxelSize );

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
  GateMessage("Geometry", 5,"GateCompressedVoxelParam::ConstructOwnPhysicalVolume " << GetPhysicalVolumeName() << " has been constructed." << G4endl;);
					        

  PushPhysicalVolume(m_pvParameterized);
	
}

void GateCompressedVoxelParam::DestroyGeometry(){
  
  if (m_parameterization) {
    delete m_parameterization;
  }
  m_parameterization=0;
  
  m_pvParameterized=0;
  GateVVolume::DestroyGeometry();
}


//---------------------------------------------------------------------------
