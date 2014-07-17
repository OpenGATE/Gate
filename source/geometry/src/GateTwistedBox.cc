/*
 * GateTwistedBox.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTwistedBox.hh"
#include "GateTwistedBoxMessenger.hh"

#include "G4TwistedBox.hh"

#include "GateTools.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh"

//-----------------------------------------------------------------------------------------------
// Constructor
GateTwistedBox::GateTwistedBox(const G4String& itsName,
                               const G4String& itsMaterialName,
                               G4double itsXLength,
                               G4double itsYLength,
                               G4double itsZLength,
                               G4double itsTwistAngle,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_twistedbox_solid(0), m_twistedbox_log(0),
  m_Length(itsXLength, itsYLength, itsZLength),
  m_twistAngle(itsTwistAngle),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateTwistedBoxMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GateTwistedBox::GateTwistedBox(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_twistedbox_solid(0), m_twistedbox_log(0),
  m_Length(1.0*cm, 1.0*cm, 1.0*cm), m_twistAngle(45.0*degree),
  m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateTwistedBoxMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GateTwistedBox::~GateTwistedBox()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new twistedbox solid and its logical volume.
// If flagUpdateOnly is set to 1, the twistedbox is updated rather than rebuilt.
G4LogicalVolume* GateTwistedBox::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_twistedbox_solid) {
      // Build mode: build the solid, then the logical volume
      m_twistedbox_solid
	= new G4TwistedBox(GetSolidName(),
                         m_twistAngle,
                         0.5*m_Length.x(),
                         0.5*m_Length.y(),
                         0.5*m_Length.z());
      m_twistedbox_log
	= new G4LogicalVolume(m_twistedbox_solid, mater, GetLogicalVolumeName(),0,0,0);

   }
   else {
     // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GateTwistedBox::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_twistedbox_log;
}

void GateTwistedBox::DestroyOwnSolidAndLogicalVolume()
{
  if (m_twistedbox_log)
    delete m_twistedbox_log;
  m_twistedbox_log = 0;

  if (m_twistedbox_solid)
    delete m_twistedbox_solid;
  m_twistedbox_solid = 0;

}

void GateTwistedBox::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: twisted box (twistedbox)" << G4endl;
  G4cout << GateTools::Indent(level) << "X length: " << G4BestUnit(m_Length.x(),"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Y length: " << G4BestUnit(m_Length.y(),"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Z length: " << G4BestUnit(m_Length.z(),"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Twist angle: " << m_twistAngle / degree << " deg" << G4endl;
}

G4double GateTwistedBox::ComputeMyOwnVolume() const
{
  return m_twistedbox_solid->GetCubicVolume();
}
