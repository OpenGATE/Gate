/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <utility>

#include <G4String.hh>
#include <G4Types.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4Box.hh>
#include <G4NistManager.hh>
#include <G4ThreeVector.hh>
#include <G4VisExtent.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSolid.hh>
#include <G4Colour.hh>
#include <G4VisAttributes.hh>

#include "GateVVolume.hh"
#include "GateTools.hh"
#include "GateTetMeshReader.hh"
#include "GateMessageManager.hh"
#include "GateDetectorConstruction.hh"  // <-- contains "theMaterialDatabase"
#include "GateMultiSensitiveDetector.hh"
#include "GateTetMeshBoxMessenger.hh"

#include "GateTetMeshBox.hh"


//----------------------------------------------------------------------------------------

GateTetMeshBox::GateTetMeshBox(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, false, depth),
  mPath(""), mUnitOfLength(mm), mAttributeMapPath(""), mAttributeMap(),
  pMessenger(new GateTetMeshBoxMessenger(this)),
  pEnvelopeSolid(nullptr), pEnvelopeLogical(nullptr), mRegionIDs(),
  mXmin(), mXmax(), mYmin(), mYmax(), mZmin(), mZmax(),
  pTetAssembly(), mPhysVolCopyNumOffset()
{
  // for now, don't accept children, to avoid overlaps with the tetrahedra
  if (acceptsChildren == true)
    GateWarning("The current TetMeshBox implementation "   \
                "doesn't support additional child volumes.");
  
  // set default material name
  GateVVolume::SetMaterialName("Vacuum");
}

//----------------------------------------------------------------------------------------

G4LogicalVolume* GateTetMeshBox::ConstructOwnSolidAndLogicalVolume(G4Material* material,
                                                                   G4bool flagUpdateOnly)
{
  // Update: Occurs when clock has changed, to trigger movement. Movement is implemented
  //         on the GateVVolume level, therefore we don't need to do anything.
  if (flagUpdateOnly == true && pEnvelopeLogical)
    return pEnvelopeLogical;

  GateMessage("Geometry", 1, "Building tetrahedral mesh..." << Gateendl);
  
  // upon construction or rebuild: read region attributes from file
  ReadAttributeMap();
  
  //-----------------------------------------------------
  // MESH CONSTRUCTION
  //-----------------------------------------------------

  // read tetrahedra from ELE file
  GateTetMeshReader fileReader(mUnitOfLength);
  std::vector<GateMeshTet> tetrahedra = fileReader.Read(mPath);

  pTetAssembly.reset(new G4AssemblyVolume);

  for (const auto& tet : tetrahedra)
    {
      G4Material* material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
      G4Colour colour = G4Colour::White();
      G4bool isVisible = true;
    
      // find attributes and set colour and material accordingly
      if (mAttributeMap.find(tet.regionID) != mAttributeMap.end())
        {
          material = mAttributeMap[tet.regionID].material;
          colour = mAttributeMap[tet.regionID].colour;
          isVisible = mAttributeMap[tet.regionID].isVisible;
        }
      else
        {
          GateWarning("Unknown region '" << tet.regionID << "', setting material to 'G4_AIR'.");
        }

      // create corresponding logical volume
      G4String logicalName = tet.solid->GetName() + "_logical"; 
      G4LogicalVolume* tetLogical = new G4LogicalVolume(tet.solid, material, logicalName);

      if (isVisible)
        {
          tetLogical->SetVisAttributes(colour);
        }
      else
        {
          tetLogical->SetVisAttributes(G4VisAttributes::GetInvisible());
        }

      // cache region marker of tetrahedron
      mRegionIDs.push_back(tet.regionID);

      // update extent of tetrahedral mesh
      const G4VisExtent& tetExtent = tet.solid->GetExtent();
      if(GetNumberOfTetrahedra() > 0)
        {
          // update
          mXmin = std::min(mXmin, tetExtent.GetXmin());
          mXmax = std::max(mXmax, tetExtent.GetXmax());
          mYmin = std::min(mYmin, tetExtent.GetYmin());
          mYmax = std::max(mYmax, tetExtent.GetYmax());
          mZmin = std::min(mZmin, tetExtent.GetZmin());
          mZmax = std::max(mZmax, tetExtent.GetZmax());
        }
      else
        {
          // init
          mXmin = tetExtent.GetXmin();
          mXmax = tetExtent.GetXmax();
          mYmin = tetExtent.GetYmin();
          mYmax = tetExtent.GetYmax();
          mZmin = tetExtent.GetZmin();
          mZmax = tetExtent.GetZmax();
        }

      // add tetrahedron to assembly, placement is trivial
      G4ThreeVector nullVector = G4ThreeVector();
      pTetAssembly->AddPlacedVolume(tetLogical, nullVector, nullptr);
    }

  //-----------------------------------------------------
  // ADAPT BOUNDING BOX & IMPRINT
  //-----------------------------------------------------

  // Create a bounding box, size is the tetrahedral mesh's extent
  G4double xHalfLength = 0.5 * (mXmax - mXmin);
  G4double yHalfLength = 0.5 * (mYmax - mYmin);
  G4double zHalfLength = 0.5 * (mZmax - mZmin);
  
  pEnvelopeSolid = new G4Box(GateVVolume::GetSolidName(),
                             xHalfLength, yHalfLength, zHalfLength);
  pEnvelopeLogical = new G4LogicalVolume(pEnvelopeSolid, material,
                                         GateVVolume::GetLogicalVolumeName());

  // place center of tetrahedral mesh at the center of the bounding box
  G4double xMean = 0.5 * (mXmax + mXmin);
  G4double yMean = 0.5 * (mYmax + mYmin);
  G4double zMean = 0.5 * (mZmax + mZmin);

  G4ThreeVector translation(-xMean, -yMean, -zMean);
  pTetAssembly->MakeImprint(pEnvelopeLogical, translation, nullptr);
  
  // ----call after imprint!!!----
  // The physical volume copy number of the first tetrahedron
  const G4VPhysicalVolume* firstPV = *(pTetAssembly->GetVolumesIterator());
  mPhysVolCopyNumOffset = firstPV->GetCopyNo();

  GateMessage("Geometry", 1, "... done building tetrahedral mesh." << Gateendl);
  return pEnvelopeLogical;
}

//----------------------------------------------------------------------------------------

void GateTetMeshBox::DestroyOwnSolidAndLogicalVolume()
{  
  // delete subtree
  if (pTetAssembly)
    {
      // manually delete logical and solid of the tetrahedra
      for(unsigned i = 0; i < pTetAssembly->TotalImprintedVolumes(); ++i)
        {
          const G4VPhysicalVolume* tetPhysical = *(pTetAssembly->GetVolumesIterator() + i);
          G4LogicalVolume* tetLogical = tetPhysical->GetLogicalVolume();
          G4VSolid* tetSolid = tetLogical->GetSolid();

          delete tetLogical;
          delete tetSolid;
        }

      // Invokes destruction of the tetrahedra's physical volumes & rotation.
      // They are owned by the assembly.
      pTetAssembly.reset(nullptr);
    }

  // delete envelope box
  if (pEnvelopeSolid)
    {
      delete pEnvelopeSolid;
      pEnvelopeSolid = nullptr;
    }
  if (pEnvelopeLogical)
    {
      delete pEnvelopeLogical;
      pEnvelopeLogical = nullptr;
    }
}

//----------------------------------------------------------------------------------------

G4double GateTetMeshBox::GetHalfDimension(size_t axis)
{
  if (pEnvelopeSolid)
    {
      const G4VisExtent extent = pEnvelopeSolid->GetExtent();
      if (axis == 0)
        {
          return 0.5 * (extent.GetXmax() - extent.GetXmin());
        } 
      else if (axis == 1)
        {
          return 0.5 * (extent.GetYmax() - extent.GetYmin());
        } 
      else if (axis == 2)
        {
          return 0.5 * (extent.GetZmax() - extent.GetZmin());
        } else
        {
          return 0.0;
        }
    }
  return 0.0;
}

//----------------------------------------------------------------------------------------

void GateTetMeshBox::PropagateSensitiveDetectorToChild(GateMultiSensitiveDetector* msd)
{
  // set sensitive detector for all daughters of the envelope box, i.e. all tetrahedra
  for (unsigned int i = 0; i < pEnvelopeLogical->GetNoDaughters(); ++i)
    {
      pEnvelopeLogical->GetDaughter(i)->GetLogicalVolume()->SetSensitiveDetector(msd);
    }
}

void GateTetMeshBox::PropagateGlobalSensitiveDetector()
{
  // in case no global SD was assigned to this volume
  if (GateVVolume::m_sensitiveDetector == nullptr)
    return;

  // otherwise check for phantom SD
  GatePhantomSD* phantomSD = \
    GateDetectorConstruction::GetGateDetectorConstruction()->GetPhantomSD();
  if (phantomSD)
    {
      // set for all tetrahedra
      for (unsigned int i = 0; i < pEnvelopeLogical->GetNoDaughters(); ++i)
        {
          pEnvelopeLogical->GetDaughter(i)->GetLogicalVolume()->SetSensitiveDetector(phantomSD);
        }
    }
}

void GateTetMeshBox::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level)
         << "From ELE file: '" << mPath << "'" << Gateendl;
  G4cout << GateTools::Indent(level)
         << "Extent: " << pEnvelopeSolid->GetExtent() << Gateendl;
  G4cout << GateTools::Indent(level)
         << "#tetrahedra: " << GetNumberOfTetrahedra() << Gateendl;
}

void GateTetMeshBox::ReadAttributeMap()
{
  GateMessage("Geometry", 2, "Reading tet attributes from file: '" <<
              mAttributeMapPath << "'." << Gateendl);
  std::ifstream inputFileStream(mAttributeMapPath);
  if (inputFileStream.is_open() == false)
    {
      GateError("Cannot open material map: '" << mAttributeMapPath << "'");
      return;
    }

  G4String line;
  while (std::getline(inputFileStream, line))
    {
      // skip comments & empty lines
      if (line.front() == '#' || line.empty())
        continue;
    
      // columns
      G4int regionIDstart;
      G4int regionIDend;
      G4String materialName;
      G4bool isVisible;
      G4double r, g, b, alpha;
    
      std::istringstream lineStream(line);

      // read columns
      if(lineStream >> regionIDstart >> regionIDend)
        if (lineStream >> materialName)
          if (lineStream >> std::boolalpha >> isVisible)
            lineStream >> r >> g >> b >> alpha;

      if (lineStream.fail())
        GateError("Failed to read line '" << line << "' in attribute map.");

      for (G4int rID = regionIDstart; rID <= regionIDend; ++rID)
        {
          GateMeshTetAttributes attributes;
          attributes.material = theMaterialDatabase.GetMaterial(materialName);
          attributes.colour = G4Colour(r, g, b, alpha);
          attributes.isVisible = isVisible;

          mAttributeMap[rID] = attributes;
        }
    }

  // print as table
  GateMessage("Geometry", 3, Gateendl);
  for (const auto& pair : mAttributeMap)
    {
      G4int regionID = pair.first;
      const GateMeshTetAttributes& attributes = pair.second;
      GateMessage("Geometry", 3, "Region " << regionID << ":\t" <<
                  attributes.material->GetName() << "\t" <<
                  attributes.colour << Gateendl);
    }
  GateMessage("Geometry", 3, Gateendl);  
}
