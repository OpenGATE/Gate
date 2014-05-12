/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateCompressedVoxelParam_h
#define GateCompressedVoxelParam_h 1

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateCompressedVoxelParameterization.hh"
#include "G4ThreeVector.hh"
#include "GateBox.hh"

class GateCompressedVoxelParameterized;
class G4VPhysicalVolume;

class GateCompressedVoxelParam : public GateBox
{
public:
  
  //! Constructor
  GateCompressedVoxelParam( const G4String& itsName, GateCompressedVoxelParameterized* vpi);
  //! Destructor
  virtual ~GateCompressedVoxelParam();
 
  void ConstructOwnPhysicalVolume(G4bool flagUpdate);
  
  void DestroyGeometry();
private:
  GateCompressedVoxelParameterized*         itsInserter;
  GateCompressedVoxelParameterization*      m_parameterization;
  G4VPhysicalVolume*                        m_pvParameterized;
};  

#endif
