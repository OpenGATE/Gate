/*
 * GateHype.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateHype.hh"
#include "GateHypeMessenger.hh"

#include "G4Hype.hh"
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
GateHype::GateHype(const G4String& itsName,
                               const G4String& itsMaterialName,
                               G4double itsInnerR,
                               G4double itsOuterR,
                               G4double itsInnerStereo,
                               G4double itsOuterStereo,
                               G4double itsZLength,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_hype_solid(0), m_hype_log(0),
  m_innerR(itsInnerR), m_outerR(itsOuterR),
  m_innerStereo(itsInnerStereo), m_outerStereo(itsOuterStereo),
  m_zLength(itsZLength),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateHypeMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GateHype::GateHype(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_hype_solid(0), m_hype_log(0),
  m_innerR(0.0*cm), m_outerR(1.0*cm),
  m_innerStereo(0*degree), m_outerStereo(0*degree),
  m_zLength(1.0*cm),
  m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateHypeMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GateHype::~GateHype()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new hype solid and its logical volume.
// If flagUpdateOnly is set to 1, the hype is updated rather than rebuilt.
G4LogicalVolume* GateHype::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_hype_solid) {
      // Build mode: build the solid, then the logical volume
      m_hype_solid
	= new G4Hype(GetSolidName(),
                         m_innerR,
                         m_outerR,
                         m_innerStereo,
                         m_outerStereo,
                         m_zLength/2.0);
      m_hype_log
	= new G4LogicalVolume(m_hype_solid, mater, GetLogicalVolumeName(),0,0,0);

   }
   else {
     // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GateHype::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_hype_log;
}

void GateHype::DestroyOwnSolidAndLogicalVolume()
{
  if (m_hype_log)
    delete m_hype_log;
  m_hype_log = 0;

  if (m_hype_solid)
    delete m_hype_solid;
  m_hype_solid = 0;

}

void GateHype::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: hyperbolic tube (hypertube)" << G4endl;
  G4cout << GateTools::Indent(level) << "Inner radius at z=0: " << G4BestUnit(m_innerR,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Outer radius at z=0: " << G4BestUnit(m_outerR,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Inner stereo angle: " << m_innerStereo / degree << " deg" << G4endl;
  G4cout << GateTools::Indent(level) << "Outer stereo angle:: " << m_outerStereo / degree << " deg" << G4endl;
  G4cout << GateTools::Indent(level) << "Z length: " << G4BestUnit(m_zLength,"Length") << G4endl;
}

G4double GateHype::ComputeMyOwnVolume() const
{
  return m_hype_solid->GetCubicVolume();
}
