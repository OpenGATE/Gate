/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GATE_TET_MESH_BOX_HH
#define GATE_TET_MESH_BOX_HH

#include <memory>
#include <map>

#include <G4String.hh>
#include <G4Types.hh>
#include <G4LogicalVolume.hh>
#include <G4VPhysicalVolume.hh>
#include <G4Colour.hh>
#include <G4Material.hh>
#include <G4Box.hh>
#include <G4AssemblyVolume.hh>

#include "GateTetMeshReader.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"

class GateTetMeshBoxMessenger;
class GateMultiSensitiveDetector;


struct GateMeshTetAttributes
{
  G4Colour colour;
  G4bool isVisible;
  G4Material* material;
};

// Integer key corresponds to identifier of a tetrahedron, i.e. region-ID.
typedef std::map<G4int, GateMeshTetAttributes> GateMeshTetAttributeMap;


// hosts a tetrahedral-mesh geometry in a box envelope
class GateTetMeshBox : public GateVVolume
{
  public:
    GateTetMeshBox(const G4String& itsName,
                   G4bool acceptsChildren = false,
                   G4int depth = 0);

    FCT_FOR_AUTO_CREATOR_VOLUME(GateTetMeshBox)
    
    // implementation of GateVVolume's interface
    G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool) final;
    void DestroyOwnSolidAndLogicalVolume() final;
    G4double GetHalfDimension(std::size_t) final;
    void PropagateSensitiveDetectorToChild(GateMultiSensitiveDetector*) final;
    void PropagateGlobalSensitiveDetector() final;

  public:
    // setters for the messenger
    void SetPathToELEFile(const G4String& path) { mPath = path; }
    void SetPathToAttributeMap(const G4String& path) { mAttributeMapPath = path; }
    void SetUnitOfLength(G4double unitOfLength) { mUnitOfLength = unitOfLength; }

    // getters for attached actors (be aware, that there is no bound checking):
    //
    std::size_t GetNumberOfTetrahedra()
    {
      return mRegionIDs.size();
    }

    // The tetrahedra are imprinted in order, as physical volumes with consecutive copy numbers.
    // However, these copy numbers may start at values > 0. This is a convenience function
    // to subtract this offset and get the tetrahedron index.
    G4int GetTetIndex(G4int physVolCopyNum) const
    {
      return physVolCopyNum - mPhysVolCopyNumOffset;
    }

    G4int GetRegionMarker(std::size_t tetIndex) const
    {
      return mRegionIDs[tetIndex];
    }

    const G4LogicalVolume* GetTetLogical(std::size_t tetIndex) const
    {
      G4VPhysicalVolume* physVol = *(pTetAssembly->GetVolumesIterator() + tetIndex);
      return physVol->GetLogicalVolume();
    }

  private:
    // implementation specifics
    void DescribeMyself(size_t);
    void ReadAttributeMap();

  private:
    G4String mPath;
    G4double mUnitOfLength;
    G4String mAttributeMapPath;
    GateMeshTetAttributeMap mAttributeMap;

    std::unique_ptr<GateTetMeshBoxMessenger> pMessenger;

    // box as mother volume containing the tetrahedra
    G4Box* pEnvelopeSolid;
    G4LogicalVolume* pEnvelopeLogical;

    // data corresponding to the actual tetrahedral mesh:
    //
    // one per tetrahedron
    std::vector<G4int> mRegionIDs;

    // extent of the tetrahedral mesh
    G4double mXmin, mXmax, mYmin, mYmax, mZmin, mZmax;

    // assembly which facilitates the imprint
    std::unique_ptr<G4AssemblyVolume> pTetAssembly;

    // copy number of the 0th tetrahedron's physical volume
    G4int mPhysVolCopyNumOffset;
};


MAKE_AUTO_CREATOR_VOLUME(TetMeshBox,GateTetMeshBox)


#endif  // GATE_TET_MESH_BOX_HH
