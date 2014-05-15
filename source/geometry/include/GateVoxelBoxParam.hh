/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVoxelBoxParam_h
#define GateVoxelBoxParam_h 1

#include "globals.hh"
#include "GateBox.hh"
#include "GateVoxelBoxParameterization.hh"
#include "G4ThreeVector.hh"

class GateVoxelBoxParameterized;
class G4VPhysicalVolume;

class GateVoxelBoxParam : public GateBox
{
public:
  
  //! Constructor
  GateVoxelBoxParam( const G4String& itsName, GateVoxelBoxParameterized* vpi);
  //! Destructor
  virtual ~GateVoxelBoxParam();
     
  void ConstructOwnPhysicalVolume(G4bool flagUpdate);
//e  void DestroyOwnPhysicalVolumes();
  void DestroyGeometry();
  
  
private:
  GateVoxelBoxParameterized* itsInserter;
  GateVoxelBoxParameterization*      m_parameterization;
  G4VPhysicalVolume*                 m_pvParameterized;
};

#endif
