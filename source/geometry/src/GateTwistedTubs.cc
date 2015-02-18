/*
 * GateTwistedTubs.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTwistedTubs.hh"
#include "GateTwistedTubsMessenger.hh"

#include "G4TwistedTubs.hh"
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
GateTwistedTubs::GateTwistedTubs(const G4String& itsName,
                               const G4String& itsMaterialName,
                               G4double itsTwistAngle,
                               G4double itsInnerR,
                               G4double itsOuterR,
                               G4double itsNegZ,
                               G4double itsPosZ,
                               G4int itsNSegment,
                               G4double itsTotalPhi,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_twistedtubs_solid(0), m_twistedtubs_log(0),
  m_twistAngle(itsTwistAngle),
  m_innerR(itsInnerR), m_outerR(itsOuterR),
  m_negZ(itsNegZ), m_posZ(itsPosZ),
  m_nSegment(itsNSegment), m_totalPhi(itsTotalPhi),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateTwistedTubsMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GateTwistedTubs::GateTwistedTubs(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_twistedtubs_solid(0), m_twistedtubs_log(0),
  m_twistAngle(45*degree),
  m_innerR(1.0*cm), m_outerR(2.0*cm),
  m_negZ(-1.0*cm), m_posZ(1.0*cm),
  m_nSegment(1), m_totalPhi(90*degree),
  m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateTwistedTubsMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GateTwistedTubs::~GateTwistedTubs()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new twistedtubs solid and its logical volume.
// If flagUpdateOnly is set to 1, the twistedtubs is updated rather than rebuilt.
G4LogicalVolume* GateTwistedTubs::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_twistedtubs_solid) {
      // Build mode: build the solid, then the logical volume
      m_twistedtubs_solid
	= new G4TwistedTubs(GetSolidName(),
                         m_twistAngle,
                         m_innerR,
                         m_outerR,
                         m_negZ,
                         m_posZ,
                         m_nSegment,
                         m_totalPhi);
      m_twistedtubs_log
	= new G4LogicalVolume(m_twistedtubs_solid, mater, GetLogicalVolumeName(),0,0,0);

   }
   else {
     // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GateTwistedTubs::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_twistedtubs_log;
}

void GateTwistedTubs::DestroyOwnSolidAndLogicalVolume()
{
  if (m_twistedtubs_log)
    delete m_twistedtubs_log;
  m_twistedtubs_log = 0;

  if (m_twistedtubs_solid)
    delete m_twistedtubs_solid;
  m_twistedtubs_solid = 0;

}

void GateTwistedTubs::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: twisted tubs (twistedtubs)" << G4endl;
  G4cout << GateTools::Indent(level) << "Lower endcap z position: " << G4BestUnit(m_negZ,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Upper endcap z position: " << G4BestUnit(m_posZ,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Inner radius at z=0: " << G4BestUnit(m_innerR,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Outer radius at z=0: " << G4BestUnit(m_outerR,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Number of segments: " << m_nSegment << G4endl;
  G4cout << GateTools::Indent(level) << "Total phi coverage: " << m_totalPhi / degree << " deg" << G4endl;
  G4cout << GateTools::Indent(level) << "Twist angle: " << m_twistAngle / degree << " deg" << G4endl;
}

G4double GateTwistedTubs::ComputeMyOwnVolume() const
{
  return m_twistedtubs_solid->GetCubicVolume();
}
