/*
 * GateTwistedTrap.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTwistedTrap.hh"
#include "GateTwistedTrapMessenger.hh"

#include "G4TwistedTrap.hh"

#include "GateTools.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh"

//-----------------------------------------------------------------------------------------------
// Constructor
GateTwistedTrap::GateTwistedTrap(const G4String& itsName,
                               const G4String& itsMaterialName,
                               G4double itsZLength,
                               G4double itsYMinusLength,
                               G4double itsYPlusLength,
                               G4double itsX1MinusLength,
                               G4double itsX2MinusLength,
                               G4double itsX1PlusLength,
                               G4double itsX2PlusLength,
                               G4double itsTwistAngle,
                               G4double itsPolarAngle,
                               G4double itsAzimuthalAngle,
                               G4double itsTiltAngle,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_twistedtrap_solid(0), m_twistedtrap_log(0),
  m_zLength(itsZLength),
  m_yMinusLength(itsYMinusLength),
  m_yPlusLength(itsYPlusLength),
  m_x1MinusLength(itsX1MinusLength),
  m_x2MinusLength(itsX2MinusLength),
  m_x1PlusLength(itsX1PlusLength),
  m_x2PlusLength(itsX2PlusLength),
  m_twistAngle(itsTwistAngle),
  m_polarAngle(itsPolarAngle),
  m_azimuthalAngle(itsAzimuthalAngle),
  m_tiltAngle(itsTiltAngle),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateTwistedTrapMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GateTwistedTrap::GateTwistedTrap(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_twistedtrap_solid(0), m_twistedtrap_log(0),
  m_zLength(1.0*cm),
  m_yMinusLength(1.0*cm),
  m_yPlusLength(1.0*cm),
  m_x1MinusLength(1.0*cm),
  m_x2MinusLength(1.0*cm),
  m_x1PlusLength(1.0*cm),
  m_x2PlusLength(1.0*cm),
  m_twistAngle(45*degree),
  m_polarAngle(45*degree),
  m_azimuthalAngle(45*degree),
  m_tiltAngle(45*degree),
  m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateTwistedTrapMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GateTwistedTrap::~GateTwistedTrap()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new twistedtrap solid and its logical volume.
// If flagUpdateOnly is set to 1, the twistedtrap is updated rather than rebuilt.
G4LogicalVolume* GateTwistedTrap::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_twistedtrap_solid) {
      // Build mode: build the solid, then the logical volume
      m_twistedtrap_solid
	= new G4TwistedTrap(GetSolidName(),
                         m_twistAngle,
                         0.5*m_zLength,
                         m_polarAngle,
                         m_azimuthalAngle,
                         0.5*m_yMinusLength,
                         0.5*m_x1MinusLength,
                         0.5*m_x2MinusLength,
                         0.5*m_yPlusLength,
                         0.5*m_x1PlusLength,
                         0.5*m_x2PlusLength,
                         m_tiltAngle);
      m_twistedtrap_log
	= new G4LogicalVolume(m_twistedtrap_solid, mater, GetLogicalVolumeName(),0,0,0);
   }
   else {
     // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GateTwistedTrap::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_twistedtrap_log;
}

void GateTwistedTrap::DestroyOwnSolidAndLogicalVolume()
{
  if (m_twistedtrap_log)
    delete m_twistedtrap_log;
  m_twistedtrap_log = 0;

  if (m_twistedtrap_solid)
    delete m_twistedtrap_solid;
  m_twistedtrap_solid = 0;

}

void GateTwistedTrap::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: twisted trapezoid (twistedtrap)\n";
  G4cout << GateTools::Indent(level) << "z length     : " << G4BestUnit(m_zLength,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "y length at -z/2   : " << G4BestUnit(m_yMinusLength,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "y length at +z/2   : " << G4BestUnit(m_yPlusLength,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "x1 length at -z/2, -y/2  : " << G4BestUnit(m_x1MinusLength,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "x2 length at -z/2, +y/2  : " << G4BestUnit(m_x2MinusLength,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "x1 length at +z/2, -y/2  : " << G4BestUnit(m_x1PlusLength,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "x2 length at +z/2, +y/2  : " << G4BestUnit(m_x2PlusLength,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "twist angle     : " << m_twistAngle/degree << " deg" << G4endl;
  G4cout << GateTools::Indent(level) << "polar angle     : " << m_polarAngle/degree << " deg" << G4endl;
  G4cout << GateTools::Indent(level) << "azimuthal angle     : " << m_azimuthalAngle/degree << " deg" << G4endl;
  G4cout << GateTools::Indent(level) << "tilt angle     : " << m_tiltAngle/degree << " deg" << G4endl;
}

G4double GateTwistedTrap::ComputeMyOwnVolume() const
{
  return m_twistedtrap_solid->GetCubicVolume();
}
