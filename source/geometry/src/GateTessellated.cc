/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "G4Material.hh"
#include "G4ThreeVector.hh"
#include "G4LogicalVolume.hh"
#include "G4TessellatedSolid.hh"
#include "G4TriangularFacet.hh"
#include "G4QuadrangularFacet.hh"

#include "GateTools.hh"

#include "GateTessellated.hh"
#include "GateTessellatedMessenger.hh"

GateTessellated::GateTessellated(G4String const &itsName, G4String const &itsMaterialName)
    : GateVVolume(itsName, false, 0),
      m_tessellated_solid(NULL),
      m_tessellated_log(NULL),
      // m_tessellated_phys( NULL ), not used
      m_PathToSTLFile(""),
      m_Messenger(NULL)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateTessellatedMessenger(this);
}

GateTessellated::GateTessellated(G4String const &itsName, G4bool itsFlagAcceptChildren, G4int depth)
    : GateVVolume(itsName, itsFlagAcceptChildren, depth),
      m_tessellated_solid(NULL),
      m_tessellated_log(NULL),
      // m_tessellated_phys( NULL ), not used
      m_PathToSTLFile(""),
      m_Messenger(NULL)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateTessellatedMessenger(this);
}

GateTessellated::~GateTessellated()
{
  delete m_Messenger;
}

G4LogicalVolume *GateTessellated::ConstructOwnSolidAndLogicalVolume(G4Material *mater, G4bool flagUpdateOnly)
{
  // Update: Occurs when clock has changed, to trigger movement. Movement is implemented
  //         on the GateVVolume level, therefore we don't need to do anything.
  if (flagUpdateOnly && m_tessellated_log)
    return m_tessellated_log;

  if (GetVerbosity() >= 2)
  {
    G4cout << "GateTessellated::ConstructOwnSolidAndLogicalVolume" << G4endl;
    DescribeMyself(1);
  }

  if (!flagUpdateOnly || !m_tessellated_solid)
  {
    // Build mode: build the solid, then the logical volume
    m_tessellated_solid = new G4TessellatedSolid(GetSolidName());

    std::ifstream STLFile(m_PathToSTLFile, std::ios::in);
    std::string line1, line2;

    if (!STLFile)
    {
      G4cerr << "No STL file: " << m_PathToSTLFile << G4endl;
    }
    else
    {
      std::getline(STLFile, line1);
      std::getline(STLFile, line2);
      STLFile.close();
      // Check the first two line contents to determine file type and read accordingly
      if ((line1.find("solid") != std::string::npos) && (line2.find("facet") != std::string::npos))
      {
        ReadSTL_ASCII();
      }
      else
      {
        ReadSTL_Binary();
      }

      m_tessellated_log = new G4LogicalVolume(m_tessellated_solid, mater, GetLogicalVolumeName());
    }
  }

  return m_tessellated_log;
}

void GateTessellated::DestroyOwnSolidAndLogicalVolume()
{
  if (m_tessellated_log)
    delete m_tessellated_log;
  m_tessellated_log = NULL;

  if (m_tessellated_solid)
    delete m_tessellated_solid;
  m_tessellated_solid = NULL;
}

void GateTessellated::SetPathToSTLFile(G4String path)
{
  m_PathToSTLFile = path;
}

void GateTessellated::ReadSTL_ASCII()
{
  G4ThreeVector v;
  std::vector<G4ThreeVector> vertices;
  std::ifstream STLFile(m_PathToSTLFile, std::ios::in);
  std::string line;
  std::istringstream vertexline;
  std::string dummy;

  nbFacets = 0;
  while (!STLFile.eof())
  {
    std::getline(STLFile, line);
    // A new facet has been found
    if (line.find("outer loop") != std::string::npos)
    {
      nbFacets++;
      std::getline(STLFile, line);
      while (line.find("vertex") != std::string::npos)
      {
        vertexline.str(line);
        vertexline.clear();
        vertexline >> dummy >> v[0] >> v[1] >> v[2];
        vertices.push_back(v);
        std::getline(STLFile, line);
      }
      // Check if the facet is correctly defined
      if (vertices.size() == 3)
      {
        FacetType = "Triangular";
        //G4cout << "New triangular facet found in the STL ASCII file" << G4endl;
        //G4cout << vertices[0] << vertices[1] << vertices[2] << G4endl;
        // Create the new facet
        G4TriangularFacet *facet = new G4TriangularFacet(vertices[0] * mm, vertices[1] * mm, vertices[2] * mm, ABSOLUTE);
        m_tessellated_solid->AddFacet((G4VFacet *)facet);
      }
      else if (vertices.size() == 4)
      {
        FacetType = "Quadrangular";
        //G4cout << "New quadrangular facet found in the STL ASCII file" << vertices.data() << G4endl;
        //G4cout << vertices[0] << vertices[1] << vertices[2] << vertices[3] << G4endl;
        // Create the new facet
        G4QuadrangularFacet *facet = new G4QuadrangularFacet(vertices[0] * mm, vertices[1] * mm, vertices[2] * mm, vertices[3] * mm, ABSOLUTE);
        m_tessellated_solid->AddFacet((G4VFacet *)facet);
      }
      else
      {
        G4cerr << "STL read error: ascii file contains unsupported number of vertices: " << vertices.size() << G4endl;
      }
      vertices.clear();
    }
  }

  STLFile.close();

  m_tessellated_solid->SetSolidClosed(true);
}

void GateTessellated::ReadSTL_Binary()
{
  char header[81];
  typedef struct
  {
    float n[3], v1[3], v2[3], v3[3];
    unsigned int att;
  } TFacet_t;
  typedef struct
  {
    float n[3], v1[3], v2[3], v3[3], v4[3];
    unsigned int att;
  } QFacet_t;

  std::ifstream STLFile(m_PathToSTLFile, std::ios::in | std::ios::binary);

  STLFile.seekg(0, std::ios::end);
  long fileSize = STLFile.tellg();
  STLFile.seekg(0, std::ios::beg);

  STLFile.read(header, 80);
  header[80] = '\0';
  STLFile.read((char *)&nbFacets, 4);

  // Check if the facets are correctly defined
  if ((int)nbFacets == (fileSize - 84) / 50)
  {
    FacetType = "Triangular";
  }
  else if ((int)nbFacets == (fileSize - 84) / 62)
  {
    FacetType = "Quadrangular";
  }
  else
  {
    G4cerr << "STL file corrupted: number of facets do not correspond to file size." << G4endl;
  }
  //G4cout << "STL file info:" << G4endl;
  //G4cout << "File size: " << fileSize << G4endl;
  //G4cout << "Header: " << header << G4endl;
  //G4cout << "Type of facets: " << FacetType << G4endl;
  //G4cout << "Number of facets: " << (int)nbFacets << G4endl;

  // Read and create all the facets
  while (true)
  {
    if (FacetType == "Triangular")
    {
      TFacet_t TFacet;
      STLFile.read((char *)&TFacet, 50);
      if (STLFile.eof())
        break;
      // Create the new facet
      G4ThreeVector vertice1 = G4ThreeVector(TFacet.v1[0], TFacet.v1[1], TFacet.v1[2]);
      G4ThreeVector vertice2 = G4ThreeVector(TFacet.v2[0], TFacet.v2[1], TFacet.v2[2]);
      G4ThreeVector vertice3 = G4ThreeVector(TFacet.v3[0], TFacet.v3[1], TFacet.v3[2]);
      G4TriangularFacet *facet = new G4TriangularFacet(vertice1 * mm, vertice2 * mm, vertice3 * mm, ABSOLUTE);
      m_tessellated_solid->AddFacet((G4VFacet *)facet);
    }
    else if (FacetType == "Quadrangular")
    {
      QFacet_t QFacet;
      STLFile.read((char *)&QFacet, 62);
      if (STLFile.eof())
        break;
      // Create the new facet
      G4ThreeVector vertice1 = G4ThreeVector(QFacet.v1[0], QFacet.v1[1], QFacet.v1[2]);
      G4ThreeVector vertice2 = G4ThreeVector(QFacet.v2[0], QFacet.v2[1], QFacet.v2[2]);
      G4ThreeVector vertice3 = G4ThreeVector(QFacet.v3[0], QFacet.v3[1], QFacet.v3[2]);
      G4ThreeVector vertice4 = G4ThreeVector(QFacet.v4[0], QFacet.v4[1], QFacet.v4[2]);
      G4QuadrangularFacet *facet = new G4QuadrangularFacet(vertice1 * mm, vertice2 * mm, vertice3 * mm, vertice4 * mm, ABSOLUTE);
      m_tessellated_solid->AddFacet((G4VFacet *)facet);
    }
  }

  STLFile.close();

  m_tessellated_solid->SetSolidClosed(true);
}

void GateTessellated::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: tessellated solid (tessellated)" << G4endl;
  G4cout << GateTools::Indent(level) << "STL file: " << m_PathToSTLFile << G4endl;
  G4cout << GateTools::Indent(level) << "Facets type: " << FacetType << G4endl;
  G4cout << GateTools::Indent(level) << "Number of facets: " << (int)nbFacets << G4endl;
}

G4double GateTessellated::ComputeMyOwnVolume() const
{
  return m_tessellated_solid->GetCubicVolume();
}
