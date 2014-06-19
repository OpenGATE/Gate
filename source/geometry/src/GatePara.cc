/*
 * GatePara.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GatePara.hh"
#include "GateParaMessenger.hh"

#include "G4Para.hh"
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
GatePara::GatePara(const G4String& itsName,
                               const G4String& itsMaterialName,
                               G4double itsXLength,
                               G4double itsYLength,
                               G4double itsZLength,
                               G4double itsAlpha,
                               G4double itsTheta,
                               G4double itsPhi,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_para_solid(0), m_para_log(0),
  m_length(itsXLength, itsYLength, itsZLength),
  m_alpha(itsAlpha), m_theta(itsTheta),
  m_phi(itsPhi),
  m_Messenger(0)
{
  SetMaterialName(itsMaterialName);
  m_Messenger = new GateParaMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Constructor with default values
GatePara::GatePara(const G4String& itsName,
                               G4bool acceptsChildren,
                               G4int depth)
: GateVVolume(itsName, acceptsChildren, depth),
  m_para_solid(0), m_para_log(0),
  m_length(1.0*cm, 1.0*cm, 1.0*cm),
  m_alpha(0*degree), m_theta(360*degree),
  m_phi(0*degree),
  m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateParaMessenger(this);
}

//-----------------------------------------------------------------------------------------------
// Destructor
GatePara::~GatePara()
{
  delete m_Messenger;
}

//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new para solid and its logical volume.
// If flagUpdateOnly is set to 1, the para is updated rather than rebuilt.
G4LogicalVolume* GatePara::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_para_solid) {
      // Build mode: build the solid, then the logical volume
      m_para_solid
	= new G4Para(GetSolidName(),
                         m_length.x()/2.0,
                         m_length.y()/2.0,
                         m_length.z()/2.0,
                         m_alpha,
                         m_theta,
                         m_phi);
      m_para_log
	= new G4LogicalVolume(m_para_solid, mater, GetLogicalVolumeName(),0,0,0);

   }
   else {
     // Update mode: refresh the dimensions of the solid
       GateMessage("Warning", 0, "GatePara::ConstructOwnSolidAndLogicalVolume update mode not implemented"<<G4endl);
   }
   return m_para_log;
}

void GatePara::DestroyOwnSolidAndLogicalVolume()
{
  if (m_para_log)
    delete m_para_log;
  m_para_log = 0;

  if (m_para_solid)
    delete m_para_solid;
  m_para_solid = 0;

}

void GatePara::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape: parallelepiped (para)" << G4endl;
  G4cout << GateTools::Indent(level) << "x length: " << G4BestUnit(m_length.x(),"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "y length: " << G4BestUnit(m_length.y(),"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "z length: " << G4BestUnit(m_length.z(),"Length") << G4endl;
  G4cout << GateTools::Indent(level) << "Alpha angle: " << m_alpha / degree << " deg" << G4endl;
  G4cout << GateTools::Indent(level) << "Polar angle: " << m_theta / degree << " deg" << G4endl;
  G4cout << GateTools::Indent(level) << "Azimuthal angle: " << m_phi / degree << " deg" << G4endl;
}

G4double GatePara::ComputeMyOwnVolume() const
{
  return m_para_solid->GetCubicVolume();
}
