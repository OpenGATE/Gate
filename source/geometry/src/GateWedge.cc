/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
  
#include "GateWedge.hh"
#include "GateWedgeMessenger.hh"

#include "G4Trap.hh"
#include "G4LogicalVolume.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"
#include "G4Material.hh"

#include "GateTools.hh"


GateWedge::GateWedge(const G4String& itsName, const G4String& itsMaterialName,
      	      	              	 G4double itsXLength, G4double itsNarrowerXLength, G4double itsYLength, G4double itsZLength)
  : GateVVolume(itsName,true,0),
    m_Wedge_solid(0), m_Wedge_log(0), 
    m_Messenger(0)
{  

  SetMaterialName(itsMaterialName);
  m_Messenger = new GateWedgeMessenger(this);
  m_WedgeLength[0] = itsXLength;
  m_WedgeLength[1] = itsNarrowerXLength;
  m_WedgeLength[2] = itsYLength;
  m_WedgeLength[3] = itsZLength;
}

GateWedge::GateWedge(const G4String& itsName,
		   G4bool itsFlagAcceptChildren,
	           G4int depth)
: GateVVolume(itsName, itsFlagAcceptChildren, depth),
  m_Wedge_solid(0), m_Wedge_log(0), m_Messenger(0) 
{
 SetMaterialName("Vacuum");
 m_WedgeLength[0] = 1.0*cm;
 m_WedgeLength[1] = 1.0*cm;
 m_WedgeLength[2] = 1.0*cm;
 m_WedgeLength[3] = 1.0*cm;
 m_Messenger = new GateWedgeMessenger(this);
}

GateWedge::~GateWedge()
{  
  if (m_Messenger)
    delete m_Messenger;
}

G4LogicalVolume* GateWedge::ConstructOwnSolidAndLogicalVolume(G4Material* mater,G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly) {
    m_Wedge_solid
      = new G4Trap(GetSolidName(), GetWedgeZLength(), GetWedgeYLength(), GetWedgeXLength(),  GetWedgeNarrowerXLength());
    m_Wedge_log
      = new G4LogicalVolume(m_Wedge_solid,mater,GetLogicalVolumeName(),0,0,0);
  }
//  else if (m_Wedge_solid) {
//    m_Wedge_solid->SetXHalfLength(GetWedgeXHalfLength());
//    m_Wedge_solid->SetYHalfLength(GetWedgeYHalfLength());
//    m_Wedge_solid->SetZHalfLength(GetWedgeZHalfLength());
//  }
  return m_Wedge_log;
}

void GateWedge::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Shape:    Trapezoidal Wedge\n";
  G4cout << GateTools::Indent(indent) << "Length along X:           " << G4BestUnit(GetWedgeXLength(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Length along narrower X:  " << G4BestUnit(GetWedgeNarrowerXLength(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Length along Y:           " << G4BestUnit(GetWedgeYLength(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Length along Z:           " << G4BestUnit(GetWedgeZLength(),"Length") << "\n";
}

void GateWedge::DestroyOwnSolidAndLogicalVolume()
{
  if (m_Wedge_log)
    delete m_Wedge_log;
  m_Wedge_log = 0;

  if (m_Wedge_solid)
    delete m_Wedge_solid;
  m_Wedge_solid = 0;

}

