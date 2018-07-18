/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GATE_TET_MESH_READER
#define GATE_TET_MESH_READER

#include <vector>

#include <G4String.hh>
#include <G4Types.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4Tet.hh>


struct GateMeshTet
{
  G4Tet* solid;

  // ELE file associate an integer attribute to each tetrahedron
  // to define to which region, i.e. "meta-shape", it belongs.
  G4int regionID;
  static constexpr G4int DEFAULT_REGION_ID = -666;
};


class GateTetMeshReader
{
  public:
    explicit GateTetMeshReader(G4double unitOfLength = mm);

    // Reads a tetrahedral mesh from a file. 
    // ELE (TetGen) is the only supported file type so far.
    std::vector<GateMeshTet> Read(const G4String& filePath);

    void SetUnitOfLength(G4double unitOfLength) { fUnitOfLength = unitOfLength; }
    G4double GetUnitOfLength() { return fUnitOfLength; }

  private:
    // implementation specifics
    std::vector<GateMeshTet> ReadELE(const G4String& filePath);
    std::vector<G4ThreeVector> ReadNODE(const G4String& filePath);
    // possible extensions, e.g.:
    // std::vecor<GateMeshTet> ReadVTKLegacy(const G4String& filePath);

  private:
    // Geant4 internal unit, used to interpret the length scale of the meshes.
    G4double fUnitOfLength;
};


#endif  // GATE_TET_MESH_READER
