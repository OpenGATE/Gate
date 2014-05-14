/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateCone.hh"
#include "GateConeMessenger.hh"
#include "GateTools.hh"

#include "G4Cons.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"
#include "G4Material.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"


//----------------------------------------------------------------------------------------------------------
GateCone::GateCone(const G4String& itsName, const G4String& /*itsMaterialName*/, //DS comment to avoid warning
      	      	   G4double itsRmax1, G4double itsRmax2, G4double itsHeight,
		   G4double itsRmin1, G4double itsRmin2, G4double itsSPhi, G4double itsDPhi,
		   G4bool itsFlagAcceptChildren, G4int depth)
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    m_cone_solid(0), m_cone_log(0),
    m_coneHeight(itsHeight),
    m_coneRmin1(itsRmin1),m_coneRmax1(itsRmax1),
  m_coneRmin2(itsRmin2),m_coneRmax2(itsRmax2),
    m_coneSPhi(itsSPhi), m_coneDPhi(itsDPhi),
  m_Messenger(0)
{
  m_Messenger = new GateConeMessenger(this);
}


//----------------------------------------------------------------------------------------------------------


GateCone::GateCone(const G4String& itsName,
                   G4bool itsFlagAcceptChildren,
                   G4int depth)
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    m_cone_solid(0), m_cone_log(0), m_Messenger(0)
{
  m_coneHeight = 1.0*cm;
  m_coneRmin1 = 1.0*cm;
  m_coneRmax1 = 1.0*cm;
  m_coneRmin2 = 1.0*cm;
  m_coneRmax2 = 1.0*cm;
  m_coneSPhi = 0.;
  m_coneDPhi = 2*M_PI;

  m_Messenger = new GateConeMessenger(this);
}
//----------------------------------------------------------------------------------------------------------


GateCone::~GateCone()
{
  delete m_Messenger;
}


//----------------------------------------------------------------------------------------------------------
G4LogicalVolume* GateCone::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{

  if (!flagUpdateOnly || !m_cone_solid) {

    m_cone_solid
      = new G4Cons(GetSolidName(),
        	   m_coneRmin1, m_coneRmax1,
      		   m_coneRmin2, m_coneRmax2,
                   m_coneHeight/2.,
                   m_coneSPhi,
                   m_coneDPhi);

    m_cone_log = new G4LogicalVolume(m_cone_solid, mater, GetLogicalVolumeName());

  }
  else {

    G4cout << " second" << G4endl;
    m_cone_solid->SetZHalfLength(GetConeHalfHeight());
    m_cone_solid->SetInnerRadiusMinusZ(GetConeRmin1());
    m_cone_solid->SetOuterRadiusMinusZ(GetConeRmax1());
    m_cone_solid->SetInnerRadiusPlusZ(GetConeRmin2());
    m_cone_solid->SetOuterRadiusPlusZ(GetConeRmax2());
    m_cone_solid->SetStartPhiAngle(GetConeSPhi());
    m_cone_solid->SetDeltaPhiAngle(GetConeDPhi());
  }

  return m_cone_log;
}
//----------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------
void GateCone::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Shape:            cone\n";
  G4cout << GateTools::Indent(indent) << "Height:           " << G4BestUnit(m_coneHeight,"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Internal radius1: " << G4BestUnit(m_coneRmin1,"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "External radius1: " << G4BestUnit(m_coneRmax1,"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Internal radius2: " << G4BestUnit(m_coneRmin2,"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "External radius2: " << G4BestUnit(m_coneRmax2,"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Start phi angle : " << m_coneSPhi / degree << " deg\n";
  G4cout << GateTools::Indent(indent) << "Phi angular span: " << m_coneDPhi / degree << " deg\n";
}
//----------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------
void GateCone::DestroyOwnSolidAndLogicalVolume()
{
  if (m_cone_log)
    delete m_cone_log;
  m_cone_log = 0;

  if (m_cone_solid)
    delete m_cone_solid;
  m_cone_solid = 0;

}
//----------------------------------------------------------------------------------------------------------
