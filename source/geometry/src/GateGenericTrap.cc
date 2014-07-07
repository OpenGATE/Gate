/*
 * GateGenericTrap.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateGenericTrap.hh"
#include "GateGenericTrapMessenger.hh"

#include "G4GenericTrap.hh"
#include "GateTools.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh"

//-----------------------------------------------------------------------------------------------
// Constructor
GateGenericTrap::GateGenericTrap(const G4String& itsName,
                               const G4String& itsMaterialName,
                               std::vector<G4TwoVector> itsVertices,
                               G4double itsZLength,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_generictrap_solid(0), m_generictrap_log(0),
  m_zLength(itsZLength),
  m_vertices(itsVertices),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateGenericTrapMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GateGenericTrap::GateGenericTrap(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_generictrap_solid(0), m_generictrap_log(0),
  m_zLength(1.0*cm),
  m_Messenger(0)
{
  m_vertices.resize(8);
  SetMaterialName("Vacuum");
  m_Messenger = new GateGenericTrapMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GateGenericTrap::~GateGenericTrap()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new generictrap solid and its logical volume.
// If flagUpdateOnly is set to 1, the generictrap is updated rather than rebuilt.
G4LogicalVolume* GateGenericTrap::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_generictrap_solid) {
      // Build mode: build the solid, then the logical volume
      m_generictrap_solid
	= new G4GenericTrap(GetSolidName(),
                         m_zLength/2.0,
                         m_vertices);
      m_generictrap_log
	= new G4LogicalVolume(m_generictrap_solid, mater, GetLogicalVolumeName(),0,0,0);

   }
   else {
     // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GateGenericTrap::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_generictrap_log;
}

void GateGenericTrap::DestroyOwnSolidAndLogicalVolume()
{
  if (m_generictrap_log)
    delete m_generictrap_log;
  m_generictrap_log = 0;

  if (m_generictrap_solid)
    delete m_generictrap_solid;
  m_generictrap_solid = 0;

}

void GateGenericTrap::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: generic trapezoid (generictrapezoid)" << G4endl;
  G4cout << GateTools::Indent(level) << "Z length: " << G4BestUnit(m_zLength,"Length") << G4endl;
  for (std::vector<G4TwoVector>::iterator i=m_vertices.begin(); i!=m_vertices.end(); ++i) {
      G4cout << GateTools::Indent(level) << "vertex at: (" << G4BestUnit((*i).x(),"Length") << ", "
							   << G4BestUnit((*i).x(),"Length") << ")" << G4endl;
  }
}

G4double GateGenericTrap::ComputeMyOwnVolume() const
{
  return m_generictrap_solid->GetCubicVolume();
}
