/*
 * GateTorus.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTorus.hh"
#include "GateTorusMessenger.hh"

#include "G4Torus.hh"
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
GateTorus::GateTorus(const G4String& itsName,
                               const G4String& itsMaterialName,
                               G4double itsInnerR,
                               G4double itsOuterR,
                               G4double itsStartPhi,
                               G4double itsDeltaPhi,
                               G4double itsTorusR,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_torus_solid(0), m_torus_log(0),
  m_innerR(itsInnerR), m_outerR(itsOuterR),
  m_startPhi(itsStartPhi), m_deltaPhi(itsDeltaPhi),
  m_torusR(itsTorusR),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateTorusMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GateTorus::GateTorus(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_torus_solid(0), m_torus_log(0),
  m_innerR(1.0*cm), m_outerR(2.0*cm),
  m_startPhi(0*degree), m_deltaPhi(360*degree),
  m_torusR(2.0*cm),
  m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateTorusMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GateTorus::~GateTorus()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new torus solid and its logical volume.
// If flagUpdateOnly is set to 1, the torus is updated rather than rebuilt.
G4LogicalVolume* GateTorus::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_torus_solid) {
      // Build mode: build the solid, then the logical volume
      m_torus_solid
	= new G4Torus(GetSolidName(),
                         m_innerR,
                         m_outerR,
                         m_torusR,
                         m_startPhi,
                         m_deltaPhi);
      m_torus_log
	= new G4LogicalVolume(m_torus_solid, mater, GetLogicalVolumeName(),0,0,0);

   }
   else {
     // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GateTorus::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_torus_log;
}

void GateTorus::DestroyOwnSolidAndLogicalVolume()
{
  if (m_torus_log)
    delete m_torus_log;
  m_torus_log = 0;

  if (m_torus_solid)
    delete m_torus_solid;
  m_torus_solid = 0;

}

void GateTorus::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: torus (torus)" << G4endl;
  G4cout << GateTools::Indent(level) << "Inner radius: " << G4BestUnit(m_innerR,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Outer radius: " << G4BestUnit(m_outerR,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Toroidal radius: " << G4BestUnit(m_torusR,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Start angle of the segment: " << m_startPhi / degree << " deg" << G4endl;
  G4cout << GateTools::Indent(level) << "Delta angle of the segment: " << m_deltaPhi / degree << " deg" << G4endl;

}

G4double GateTorus::ComputeMyOwnVolume() const
{
  return m_torus_solid->GetCubicVolume();
}
