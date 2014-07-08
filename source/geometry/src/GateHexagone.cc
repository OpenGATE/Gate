/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateHexagone.hh"
#include "GateHexagoneMessenger.hh"

#include "GatePolyhedra.hh"
#include "G4LogicalVolume.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"


GateHexagone::GateHexagone(const G4String& itsName, const G4String& itsMaterialName,
      	      	      	      	      	 G4double itsRadius,G4double itsHeight)
  : GateVVolume(itsName,itsMaterialName),
    m_HexagoneRadius(itsRadius),m_HexagoneHeight(itsHeight),
    m_HexagoneZPlane(0), m_HexagoneRInner(0), m_HexagoneROuter(0), 
    m_Hexagone_solid(0), m_Hexagone_log(0),

    m_Messenger(0)
{
    SetMaterialName(itsMaterialName);
    m_Messenger = new GateHexagoneMessenger(this);
  
    m_HexagoneZPlane = new G4double[2];
    m_HexagoneRInner = new G4double[6];
    m_HexagoneROuter = new G4double[6];
  
}


GateHexagone::GateHexagone(const G4String& itsName,
		           G4bool itsFlagAcceptChildren,
			   G4int depth)
: GateVVolume(itsName, itsFlagAcceptChildren, depth),
  m_Hexagone_solid(0), m_Hexagone_log(0), m_Messenger(0) 
{
     // Default material name
     SetMaterialName("Vacuum");
     
     m_HexagoneZPlane = new G4double[2];
     m_HexagoneRInner = new G4double[6];
     m_HexagoneROuter = new G4double[6];
  
  
     m_Messenger = new GateHexagoneMessenger(this);
}



GateHexagone::~GateHexagone()
{  
  delete m_Messenger;
/* 
  delete[]m_HexagoneZPlane;
  delete[]m_HexagoneRInner;
  delete[]m_HexagoneROuter;
*/  
}



G4LogicalVolume* GateHexagone::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly)
  {
    size_t i;
    for (i=0; i<2; i++) 
    {
      m_HexagoneZPlane[i] = i*m_HexagoneHeight-m_HexagoneHeight/2;
    }
    for (i=0; i<6; i++) 
    {
      m_HexagoneRInner[i] = 0;
      m_HexagoneROuter[i] = m_HexagoneRadius;
    }
 
  
    m_Hexagone_solid
      = new GatePolyhedra(GetSolidName(), 0, 360*deg, 6, 2, m_HexagoneZPlane,  m_HexagoneRInner, m_HexagoneROuter);
    
    m_Hexagone_log
      = new G4LogicalVolume(m_Hexagone_solid, mater,GetLogicalVolumeName(),0,0,0);
  }
  
  return m_Hexagone_log;
}

void GateHexagone::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Shape:          Hexagone\n";
  G4cout << GateTools::Indent(indent) << "Radius: " << G4BestUnit(GetHexagoneRadius(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Height: " << G4BestUnit(GetHexagoneHeight(),"Length") << "\n";
}

void GateHexagone::DestroyOwnSolidAndLogicalVolume()
{
  if (m_Hexagone_log)
    delete m_Hexagone_log;
  m_Hexagone_log = 0;

  if (m_Hexagone_solid)
    delete m_Hexagone_solid;
  m_Hexagone_solid = 0;

}

