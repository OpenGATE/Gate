/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include <string>
#include <fstream>
#include <sstream>
#include <array>
#include <vector>

#include <G4String.hh>
#include <G4Types.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4Tet.hh>

#include "GateTools.hh"
#include "GateMessageManager.hh"

#include "GateTetMeshReader.hh"


//----------------------------------------------------------------------------------------

GateTetMeshReader::GateTetMeshReader(G4double unitOfLength)
  : fUnitOfLength(unitOfLength)
{
}

//----------------------------------------------------------------------------------------

std::vector<GateMeshTet> GateTetMeshReader::Read(const G4String& filePath)
{
  std::vector<GateMeshTet> tetrahedra;

  const G4String& extension = GateTools::PathSplitExt(filePath).second;
  if (extension == ".ele")
  {
    tetrahedra = ReadELE(filePath);
  }
  else
  {
    GateError("File format not supported: '" << extension << "'. Could not load tetrahedral mesh.");
  }

  return tetrahedra;
}

//----------------------------------------------------------------------------------------

std::vector<GateMeshTet> GateTetMeshReader::ReadELE(const G4String& filePath)
{
  // ELE files are accompanied by seperate NODE files which define all mesh nodes.
  // E.g. for "<filePath>.ele" there should be "<filePath>.node".
  G4String nodeFilePath = GateTools::PathSplitExt(filePath).first + ".node";
  std::vector<G4ThreeVector> nodes = ReadNODE(nodeFilePath);

  // Only after successfully reading the nodes, the ELE file is looked into.
  GateMessage("Geometry", 2, "Reading tetrahedra from '" << filePath << "'." << Gateendl);
  std::ifstream eleFileStream(filePath);
  if (eleFileStream.is_open() == false)
  {
    GateError("Cannot open file: '" << filePath << "'.");
    return std::vector<GateMeshTet>();
  }

  // The first non-comment line should be the header, containing:
  //
  //    <# of tetrahedra> <nodes per tet. (4 or 10)> <region attribute (0 or 1)>
  //
  std::size_t nTetrahedra = 0;
  std::size_t nNodesPerTet = 0;
  G4bool hasRegionID = false;

  G4String line;
  while (std::getline(eleFileStream, line))
  {
    // skip comments & emtpy lines
    if (line.front() == '#' || line.empty())
      continue;
    
    std::istringstream lineParser(line);

    // <# of tetrahedra> <nodes per tet. (4 or 10)> <region attribute (0 or 1)>
    if (lineParser >> nTetrahedra >> nNodesPerTet)
      lineParser >> hasRegionID;

    if (lineParser.fail())
    {
      GateError("Failed to parse ELE section header: '" << line << "'.");
      return std::vector<GateMeshTet>();
    }

    break;
  }

  // If generated with the '-o2' flag, TetGen's ELE file might define tetrahedra
  // consisting of 10 points, i.e. the corners and each edge's midpoint.
  // For now, abort when encountering these files.
  if (nNodesPerTet != 4)
  {
    GateError("Cannot read tetrahedral mesh generated with '-o2' flag.");
    return std::vector<GateMeshTet>();
  }

  // strings we'll need to name the solids later on
  const G4String& fileName = GateTools::PathSplit(filePath).second;
  const G4String& fileNameRoot = GateTools::PathSplitExt(fileName).first;

  // After the header, each row of the ELE file defines one tetrahedron, 
  // via the indices of specific nodes: 
  //    ...
  //    <tetrahedron #> <node> <node> ... <node> [attribute]
  //    ...
  std::vector<GateMeshTet> tetrahedra;
  std::size_t counter = 0;
  while (std::getline(eleFileStream, line) && counter < nTetrahedra)
  {
    // skip comments & emtpy lines
    if (line.front() == '#' || line.empty())
      continue;

    std::istringstream lineParser(line);

    // <tetrahedron #>
    std::size_t tetNumber = 0;
    lineParser >> tetNumber;

    // <node> <node> ... <node>
    std::array<G4ThreeVector, 4> cornerNodes;
    for (auto& cornerNode : cornerNodes)
    {
      G4int index;
      lineParser >> index;
      cornerNode = nodes[index];
    }

    // [attribute] aka. regionID
    G4int regionID = GateMeshTet::DEFAULT_REGION_ID;
    if (hasRegionID)
      lineParser >> regionID;

    // check, if parsing any of the integers has failed
    if (lineParser.fail())
    {
      GateError("Failed to read tetrahedron: '" << line << "'.");
      return std::vector<GateMeshTet>();
    }

    G4String tetSolidName = fileNameRoot + "_tet" + std::to_string(counter);
    G4Tet* tetSolid = new G4Tet(tetSolidName, cornerNodes[0], cornerNodes[1],
                                              cornerNodes[2], cornerNodes[3]);

    tetrahedra.push_back(GateMeshTet{tetSolid, regionID});
    ++counter;
  }

  GateMessage("Geometry", 2, "Obtained mesh containting "
                             << tetrahedra.size() <<
                             " tetrahedra." << Gateendl);
  return tetrahedra;
}

//----------------------------------------------------------------------------------------

std::vector<G4ThreeVector> GateTetMeshReader::ReadNODE(const G4String& filePath)
{
  GateMessage("Geometry", 2, "Reading nodes from '" << filePath << "'." << Gateendl);
  std::ifstream nodeFileStream(filePath);
  if (nodeFileStream.is_open() == false)
  {
    GateError("Cannot open file: '" << filePath << "'.");
    return std::vector<G4ThreeVector>();
  }
  
  // header of node section:
  //    <# of nodes> <dimension (3)> <# of attributes> <boundary markers (0 or 1)>
  std::size_t nNodes = 0;
  std::size_t dimension = 0;

  G4String line;
  while (std::getline(nodeFileStream, line))
  {
    // skip comments & emtpy lines
    if (line.front() == '#' || line.empty())
      continue;

    std::istringstream lineParser(line);

    // read first entry: <# of points>
    lineParser >> nNodes;
    if (lineParser.fail())
    {
      GateError("Failed to parse node section header: '" << line << "'.");
      return std::vector<G4ThreeVector>();
    }

    if (nNodes > 0)
    {
      // read second entry: <dimension (3)>
      lineParser >> dimension;
      if (lineParser.fail() || dimension != 3)
      {
        GateError("Failed to parse node section header: '" << line << "'.");
        return std::vector<G4ThreeVector>();
      }
    }
    else
    {
      GateError("NODE file is emtpy or #nodes is set to zero.");
      return std::vector<G4ThreeVector>();
    }

    // ignore the rest of the header line, 
    // i.e. ... <# of attributes> <boundary markers (0 or 1)>
    break;
  }

  //  remaining lines list nodes: 
  //    <node #> <x> <y> <z> [attributes] [boundary marker]
  std::vector<G4ThreeVector> nodes;
  while (std::getline(nodeFileStream, line) && nodes.size() < nNodes)
  {
    // skip comments & emtpy lines
    if (line.front() == '#' || line.empty())
      continue;

    std::istringstream lineParser(line);

    std::size_t nodeCount;
    G4double x, y, z;

    // read node: <point #> <x> <y> <z>
    if (lineParser >> nodeCount)
      lineParser >> x >> y >> z;
    
    if (lineParser.fail())
    {
      GateError("Failed to read node: '" << line << "'.");
      return std::vector<G4ThreeVector>();
    }

    nodes.push_back(G4ThreeVector(x, y, z) * fUnitOfLength);
  }

  return nodes;
}
