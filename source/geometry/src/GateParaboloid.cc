/*
 * GateParaboloid.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateParaboloid.hh"
#include "GateParaboloidMessenger.hh"

#include "G4Paraboloid.hh"
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
GateParaboloid::GateParaboloid(const G4String& itsName,
                               const G4String& itsMaterialName,
                               G4double itsNegR,
                               G4double itsPosR,
                               G4double itsZLength,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_paraboloid_solid(0), m_paraboloid_log(0),
  m_negR(itsNegR), m_posR(itsPosR),
  m_zLength(itsZLength),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateParaboloidMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GateParaboloid::GateParaboloid(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_paraboloid_solid(0), m_paraboloid_log(0),
  m_negR(1.0*cm), m_posR(2.0*cm),
  m_zLength(1.0*cm),
  m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateParaboloidMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GateParaboloid::~GateParaboloid()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new paraboloid solid and its logical volume.
// If flagUpdateOnly is set to 1, the paraboloid is updated rather than rebuilt.
G4LogicalVolume* GateParaboloid::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_paraboloid_solid) {
      // Build mode: build the solid, then the logical volume
      m_paraboloid_solid
	= new G4Paraboloid(GetSolidName(),
                         m_negR,
                         m_posR,
                         m_zLength/2.0);
      m_paraboloid_log
	= new G4LogicalVolume(m_paraboloid_solid, mater, GetLogicalVolumeName(),0,0,0);

   }
   else {
     // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GateParaboloid::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_paraboloid_log;
}

void GateParaboloid::DestroyOwnSolidAndLogicalVolume()
{
  if (m_paraboloid_log)
    delete m_paraboloid_log;
  m_paraboloid_log = 0;

  if (m_paraboloid_solid)
    delete m_paraboloid_solid;
  m_paraboloid_solid = 0;

}

void GateParaboloid::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: paraboloidrbolic tube (paraboloidrtube)" << G4endl;
  G4cout << GateTools::Indent(level) << "Radius at -z/2: " << G4BestUnit(m_negR,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Radius at +z/2: " << G4BestUnit(m_posR,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Z length: " << G4BestUnit(m_zLength,"Length") << G4endl;
}

G4double GateParaboloid::ComputeMyOwnVolume() const
{
  return m_paraboloid_solid->GetCubicVolume();
}
