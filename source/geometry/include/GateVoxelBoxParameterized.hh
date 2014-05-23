/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVoxelBoxParameterized_h
#define GateVoxelBoxParameterized_h 1

#include "globals.hh"
#include "GateBox.hh"
//#include "GateVParameterisedInserter.hh"
#include "GateVoxelBoxParameterizedMessenger.hh"
#include "GateVGeometryVoxelReader.hh"
#include "GateVoxelBoxParameterization.hh"
#include "G4PVParameterised.hh"
#include "GateObjectRepeaterList.hh"
#include "GateVolumePlacement.hh"
#include "GateVoxelBoxParam.hh"

class GateVoxelBoxParameterized : public GateBox
{
public:
  // Constructor1
  GateVoxelBoxParameterized(const G4String& name,
  				    G4bool acceptsChildren=true, 
		 		    G4int depth=0)
  : GateBox(name,"Vacuum",1,1,1,acceptsChildren,depth),
    m_name(name),
    m_messenger(new GateVoxelBoxParameterizedMessenger(this)),
    m_voxelReader(0),
    m_voxelInserter(new GateVoxelBoxParam(name+"Voxel", this)),
    voxelNumber(G4ThreeVector(1,1,1)),
    voxelSize(G4ThreeVector(1,1,1))
  { 
    GetCreator()->GetTheChildList()->AddChild(m_voxelInserter);
  }

  // Constructor2
  GateVoxelBoxParameterized(const G4String& name):GateBox(name,"Vacuum",1,1,1,false,false),
							  m_name(name),
							  m_messenger(new GateVoxelBoxParameterizedMessenger(this)),
							  m_voxelReader(0),
							  m_voxelInserter(new GateVoxelBoxParam(name+"Voxel", this)),
							  voxelNumber(G4ThreeVector(1,1,1)),
							  voxelSize(G4ThreeVector(1,1,1))
  { 
    GetCreator()->GetTheChildList()->AddChild(m_voxelInserter);
  }
  
  // Destructor
  ~GateVoxelBoxParameterized(){
    delete m_messenger;
  }
  
  FCT_FOR_AUTO_CREATOR_VOLUME(GateVoxelBoxParameterized)

  //! Insert the reader to read and store the voxel material info
  void InsertReader(G4String readerType);

  //! Remove the reader to read and store the voxel material info
  void RemoveReader();

  //! Get the reader to read and store the voxel material info
  inline GateVGeometryVoxelReader* GetReader() const
  { return m_voxelReader;}
  
  //! Attach the matrix to the phantom SD
  void AttachPhantomSD();

  void AddOutput(G4String name);

  void ConstructGeometry(G4LogicalVolume* mother_log, G4bool flagUpdateOnly);


private:
  
  G4String m_name;

  GateVoxelBoxParameterizedMessenger* m_messenger;
  GateVGeometryVoxelReader*                   m_voxelReader;
  GateVoxelBoxParam*                  m_voxelInserter;
  
  G4ThreeVector                               voxelNumber;
  G4ThreeVector                               voxelSize;

  };

MAKE_AUTO_CREATOR_VOLUME(parameterizedBoxMatrix,GateVoxelBoxParameterized)

#endif
