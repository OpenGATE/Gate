/*
 * GateTetra.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTetra.hh"
#include "GateTetraMessenger.hh"

#include "GateTools.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh"

//-----------------------------------------------------------------------------------------------
// Constructor
GateTetra::GateTetra(const G4String& itsName,
                               const G4String& itsMaterialName,
                               G4ThreeVector itsP1,
                               G4ThreeVector itsP2,
                               G4ThreeVector itsP3,
                               G4ThreeVector itsP4,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_tetra_solid(0), m_tetra_log(0),
  m_p1(itsP1),m_p2(itsP2),m_p3(itsP3),m_p4(itsP4),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateTetraMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GateTetra::GateTetra(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_tetra_solid(0), m_tetra_log(0),
  m_p1(0.0*cm, 0.0*cm, 1.0*cm),
  m_p2(1.0*cm, 0.0*cm, 0.0*cm),
  m_p3(0.0*cm, 1.0*cm, 0.0*cm),
  m_p4(0.0*cm, 0.0*cm, 0.0*cm),
  m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateTetraMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GateTetra::~GateTetra()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new twistedbox solid and its logical volume.
// If flagUpdateOnly is set to 1, the twistedbox is updated rather than rebuilt.
G4LogicalVolume* GateTetra::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_tetra_solid) {
      // Build mode: build the solid, then the logical volume
      m_tetra_solid
	= new G4Tet(GetSolidName(),
                         m_p1,
                         m_p2,
                         m_p3,
                         m_p4);

      m_tetra_log
	= new G4LogicalVolume(m_tetra_solid, mater, GetLogicalVolumeName(),0,0,0);
   }
   else {
       // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GateTetra::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_tetra_log;
}

void GateTetra::DestroyOwnSolidAndLogicalVolume()
{
  if (m_tetra_log)
    delete m_tetra_log;
  m_tetra_log = 0;

  if (m_tetra_solid)
    delete m_tetra_solid;
  m_tetra_solid = 0;

}

void GateTetra::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: tetrahedron (tetra)" << G4endl;
  G4cout << GateTools::Indent(level) << "Vertex #1: " << G4BestUnit(m_p1,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Vertex #2: " << G4BestUnit(m_p2,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Vertex #3: " << G4BestUnit(m_p3,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Vertex #4: " << G4BestUnit(m_p4,"Length") << G4endl;
}

G4double GateTetra::ComputeMyOwnVolume() const
{
  return m_tetra_solid->GetCubicVolume();
}
