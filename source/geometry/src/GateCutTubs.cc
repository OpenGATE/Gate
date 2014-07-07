/*
 * GateCutTubs.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateCutTubs.hh"
#include "GateCutTubsMessenger.hh"

#include "G4CutTubs.hh"
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
GateCutTubs::GateCutTubs(const G4String& itsName,
                               const G4String& itsMaterialName,
                               G4double itsInnerR,
                               G4double itsOuterR,
                               G4double itsStartPhi,
                               G4double itsDeltaPhi,
                               G4double itsZLength,
                               G4ThreeVector itsNegNorm,
                               G4ThreeVector itsPosNorm,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_cuttubs_solid(0), m_cuttubs_log(0),
  m_innerR(itsInnerR), m_outerR(itsOuterR),
  m_startPhi(itsStartPhi), m_deltaPhi(itsDeltaPhi),
  m_zLength(itsZLength),
  m_negNorm(itsNegNorm),m_posNorm(itsPosNorm),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateCutTubsMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GateCutTubs::GateCutTubs(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_cuttubs_solid(0), m_cuttubs_log(0),
  m_innerR(1.0*cm), m_outerR(2.0*cm),
  m_startPhi(0*degree), m_deltaPhi(360*degree),
  m_zLength(2.0*cm),
  m_negNorm(0,0,-1), m_posNorm(0,0,1),
  m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateCutTubsMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GateCutTubs::~GateCutTubs()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new cuttubs solid and its logical volume.
// If flagUpdateOnly is set to 1, the cuttubs is updated rather than rebuilt.
G4LogicalVolume* GateCutTubs::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_cuttubs_solid) {
      // Build mode: build the solid, then the logical volume
      m_cuttubs_solid
	= new G4CutTubs(GetSolidName(),
                         m_innerR,
                         m_outerR,
                         m_zLength,
                         m_startPhi,
                         m_deltaPhi,
                         m_negNorm,
                         m_posNorm);
      m_cuttubs_log
	= new G4LogicalVolume(m_cuttubs_solid, mater, GetLogicalVolumeName(),0,0,0);

   }
   else {
     // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GateCutTubs::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_cuttubs_log;
}

void GateCutTubs::DestroyOwnSolidAndLogicalVolume()
{
  if (m_cuttubs_log)
    delete m_cuttubs_log;
  m_cuttubs_log = 0;

  if (m_cuttubs_solid)
    delete m_cuttubs_solid;
  m_cuttubs_solid = 0;

}

void GateCutTubs::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: cutted tube (cuttube)" << G4endl;
  G4cout << GateTools::Indent(level) << "Inner radius: " << G4BestUnit(m_innerR,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Outer radius: " << G4BestUnit(m_outerR,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Z length: " << G4BestUnit(m_zLength,"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Start angle of the segment: " << m_startPhi / degree << " deg" << G4endl;
  G4cout << GateTools::Indent(level) << "Delta angle of the segment: " << m_deltaPhi / degree << " deg" << G4endl;
  G4cout << GateTools::Indent(level) << "Cutting plane normal at -z/2: " << m_negNorm << G4endl;
  G4cout << GateTools::Indent(level) << "Cutting plane normal at +z/2: " << m_posNorm << G4endl;
}

G4double GateCutTubs::ComputeMyOwnVolume() const
{
  return m_cuttubs_solid->GetCubicVolume();
}
