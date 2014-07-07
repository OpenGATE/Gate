/*
 * GateEllipticalCone.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateEllipticalCone.hh"
#include "GateEllipticalConeMessenger.hh"

#include "G4EllipticalCone.hh"
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
GateEllipticalCone::GateEllipticalCone(const G4String& itsName,
                               const G4String& itsMaterialName,
                               G4double itsXSemiAxis,
                               G4double itsYSemiAxis,
                               G4double itsZLength,
                               G4double itsZCut,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_ellipticalcone_solid(0), m_ellipticalcone_log(0),
  m_xSemiAxis(itsXSemiAxis), m_ySemiAxis(itsYSemiAxis),
  m_zLength(itsZLength), m_zCut(itsZCut),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateEllipticalConeMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GateEllipticalCone::GateEllipticalCone(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_ellipticalcone_solid(0), m_ellipticalcone_log(0),
  m_xSemiAxis(1.0*cm), m_ySemiAxis(2.0*cm),
  m_zLength(1.0*cm), m_zCut(1.0*cm),
  m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateEllipticalConeMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GateEllipticalCone::~GateEllipticalCone()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new ellipticalcone solid and its logical volume.
// If flagUpdateOnly is set to 1, the ellipticalcone is updated rather than rebuilt.
G4LogicalVolume* GateEllipticalCone::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_ellipticalcone_solid) {
      // Build mode: build the solid, then the logical volume
      m_ellipticalcone_solid
	= new G4EllipticalCone(GetSolidName(),
                         m_xSemiAxis/(m_zLength+m_zCut)*2.0,
                         m_ySemiAxis/(m_zLength+m_zCut)*2.0,
                         m_zLength/2.0,
                         m_zCut/2.0);
      m_ellipticalcone_log
	= new G4LogicalVolume(m_ellipticalcone_solid, mater, GetLogicalVolumeName(),0,0,0);

   }
   else {
     // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GateEllipticalCone::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_ellipticalcone_log;
}

void GateEllipticalCone::DestroyOwnSolidAndLogicalVolume()
{
  if (m_ellipticalcone_log)
    delete m_ellipticalcone_log;
  m_ellipticalcone_log = 0;

  if (m_ellipticalcone_solid)
    delete m_ellipticalcone_solid;
  m_ellipticalcone_solid = 0;

}

void GateEllipticalCone::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: elliptical cone (ellipticalcone)" << G4endl;
  G4cout << GateTools::Indent(level) << "x semiaxis at the bottom of the cone: " << G4BestUnit(m_xSemiAxis,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "y semiaxis at the bottom of the cone: " << G4BestUnit(m_ySemiAxis,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Total Z height: " << G4BestUnit(m_zLength,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Height of the cutted region: " << G4BestUnit(m_zCut,"Length") << G4endl;
}

G4double GateEllipticalCone::ComputeMyOwnVolume() const
{
  return m_ellipticalcone_solid->GetCubicVolume();
}
