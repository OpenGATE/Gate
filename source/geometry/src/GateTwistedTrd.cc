/*
 * GateTwistedTrd.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTwistedTrd.hh"
#include "GateTwistedTrdMessenger.hh"

#include "G4TwistedTrd.hh"

#include "GateTools.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh"

//-----------------------------------------------------------------------------------------------
// Constructor
GateTwistedTrd::GateTwistedTrd(const G4String& itsName,
                               const G4String& itsMaterialName,
                               G4double itsX1Length,
                               G4double itsX2Length,
                               G4double itsY1Length,
                               G4double itsY2Length,
                               G4double itsZLength,
                               G4double itsTwistAngle,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_twistedtrd_solid(0), m_twistedtrd_log(0),
  m_X1Length(itsX1Length), m_X2Length(itsX2Length),
  m_Y1Length(itsY1Length), m_Y2Length(itsY2Length),
  m_ZLength(itsZLength), m_twistAngle(itsTwistAngle),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateTwistedTrdMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GateTwistedTrd::GateTwistedTrd(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_twistedtrd_solid(0), m_twistedtrd_log(0),
  m_X1Length(1.0*cm), m_X2Length(1.0*cm),
  m_Y1Length(1.0*cm), m_Y2Length(1.0*cm),
  m_ZLength(1.0*cm), m_twistAngle(45.0*degree),
  m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateTwistedTrdMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GateTwistedTrd::~GateTwistedTrd()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new twistedtrd solid and its logical volume.
// If flagUpdateOnly is set to 1, the twistedtrd is updated rather than rebuilt.
G4LogicalVolume* GateTwistedTrd::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_twistedtrd_solid) {
      // Build mode: build the solid, then the logical volume
      m_twistedtrd_solid
	= new G4TwistedTrd(GetSolidName(),
                         0.5*m_X1Length,
                         0.5*m_X2Length,
                         0.5*m_Y1Length,
                         0.5*m_Y1Length,
                         0.5*m_ZLength,
                         m_twistAngle);
      m_twistedtrd_log
	= new G4LogicalVolume(m_twistedtrd_solid, mater, GetLogicalVolumeName(),0,0,0);

   }
   else {
     // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GateTwistedTrd::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_twistedtrd_log;
}

void GateTwistedTrd::DestroyOwnSolidAndLogicalVolume()
{
  if (m_twistedtrd_log)
    delete m_twistedtrd_log;
  m_twistedtrd_log = 0;

  if (m_twistedtrd_solid)
    delete m_twistedtrd_solid;
  m_twistedtrd_solid = 0;

}

void GateTwistedTrd::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: twisted trapezoid (twistedtrd)\n";
  G4cout << GateTools::Indent(level) << "X1 length: " << G4BestUnit(m_X1Length,"Length") << "\n";
  G4cout << GateTools::Indent(level) << "X2 length: " << G4BestUnit(m_X2Length,"Length") << "\n";
  G4cout << GateTools::Indent(level) << "Y1 length: " << G4BestUnit(m_Y1Length,"Length") << "\n";
  G4cout << GateTools::Indent(level) << "Y2 length: " << G4BestUnit(m_Y2Length,"Length") << "\n";
  G4cout << GateTools::Indent(level) << "Z length: " << G4BestUnit(m_ZLength,"Length") << "\n";
  G4cout << GateTools::Indent(level) << "Twist angle: " << m_twistAngle / degree << " deg\n";
}

G4double GateTwistedTrd::ComputeMyOwnVolume() const
{
  return m_twistedtrd_solid->GetCubicVolume();
}
