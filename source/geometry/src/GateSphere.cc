/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateSphere.hh"
#include "GateSphereMessenger.hh"
#include "GateTools.hh"

#include "G4Sphere.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh"
#include "G4VisAttributes.hh"


//-----------------------------------------------------------------------------------------------
// Constructor
GateSphere::GateSphere(const G4String& itsName,const G4String& /*itsMaterialName */,
                       G4double itsRmax,
                       G4double itsRmin,
                       G4double itsSPhi, G4double itsDPhi,
                       G4double itsSTheta, G4double itsDTheta,
                       G4bool itsFlagAcceptChildren, G4int depth)
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
  m_sphere_solid(0),m_sphere_log(0),
  m_sphereRmin(itsRmin),m_sphereRmax(itsRmax),
    m_sphereSPhi(itsSPhi), m_sphereDPhi(itsDPhi),
  m_sphereSTheta(itsSTheta), m_sphereDTheta(itsDTheta),
  m_Messenger(0)
{
  m_Messenger = new GateSphereMessenger(this);
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
// Constructor
GateSphere::GateSphere(const G4String& itsName,
                       G4bool itsFlagAcceptChildren,
                       G4int depth)
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    m_sphere_solid(0),m_sphere_log(0),
    m_Messenger(0)
{
  m_sphereRmin = 0. *cm;
  m_sphereRmax = 1. *cm;
  m_sphereSPhi = 0.;
  m_sphereDPhi = 2*M_PI;
  m_sphereSTheta = 0.;
  m_sphereDTheta = M_PI;

  m_Messenger = new GateSphereMessenger(this);
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
// Destructor
GateSphere::~GateSphere()
{
  delete m_Messenger;
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
// Construct a new cylinder-solid and its logical volume.
// If flagUpdateOnly is set to 1, the cylinder is updated rather than rebuilt.
G4LogicalVolume* GateSphere::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !m_sphere_solid) {
    // Build mode: build the solid, then the logical volume
    m_sphere_solid
      = new G4Sphere(GetSolidName(),
		     m_sphereRmin, m_sphereRmax,
		     m_sphereSPhi, m_sphereDPhi,
		     m_sphereSTheta, m_sphereDTheta);
    m_sphere_log
      = new G4LogicalVolume(m_sphere_solid, mater, GetLogicalVolumeName(), 0, 0, 0);
  }
  else {
    // Update mode: refresh the dimensions of the solid
    m_sphere_solid->SetInsideRadius( m_sphereRmin);
    m_sphere_solid->SetOuterRadius(m_sphereRmax);
    m_sphere_solid->SetStartPhiAngle(m_sphereSPhi);
    m_sphere_solid->SetDeltaPhiAngle(m_sphereDPhi);
    m_sphere_solid->SetStartThetaAngle(m_sphereSTheta);
    m_sphere_solid->SetDeltaThetaAngle(m_sphereDTheta);
  }

  // To visualisation of the sphere
  m_own_visAtt = new G4VisAttributes();
  m_own_visAtt->SetForceAuxEdgeVisible(true);
  m_sphere_log->SetVisAttributes(m_own_visAtt);

  return m_sphere_log;
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
// Implementation of the pure virtual method DestroyOwnSolidAndVolume() declared by the base-class.
// Destroy the solid and logical volume created by ConstructOwnSolidAndLogical()
void GateSphere::DestroyOwnSolidAndLogicalVolume()
{
  if (m_sphere_log)
    delete m_sphere_log;
  m_sphere_log = 0;

  if (m_sphere_solid)
    delete m_sphere_solid;
  m_sphere_solid = 0;

}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
/* Implementation of the virtual method DescribeMyself(), to print-out a description of the creator

   indent: the print-out indentation (cosmetic parameter)
*/
void GateSphere::DescribeMyself(size_t level)
{
  G4cout << GateTools::Indent(level) << "Shape:              sphere\n";
  G4cout << GateTools::Indent(level) << "Internal radius:    " << G4BestUnit(m_sphereRmin,"Length") << "\n";
  G4cout << GateTools::Indent(level) << "External radius:    " << G4BestUnit(m_sphereRmax,"Length") << "\n";
  G4cout << GateTools::Indent(level) << "Start phi angle:    " << m_sphereSPhi / degree << " deg\n";
  G4cout << GateTools::Indent(level) << "Phi angular span:   " << m_sphereDPhi / degree << " deg\n";
  G4cout << GateTools::Indent(level) << "Start theta angle:  " << m_sphereSTheta / degree << " deg\n";
  G4cout << GateTools::Indent(level) << "Theta angular span: " << m_sphereDTheta / degree << " deg\n";
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
// Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
// Returns the volume of the solid
G4double GateSphere::ComputeMyOwnVolume() const
{
  return m_sphere_solid->GetCubicVolume();
}
//-----------------------------------------------------------------------------------------------
