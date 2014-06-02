/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateTrpd.hh"
#include "GateTrpdMessenger.hh"
#include "GateClock.hh"
#include "GateTools.hh"

#include "G4Trd.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"
#include "G4PVPlacement.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "G4Transform3D.hh"

GateTrpd::GateTrpd(const G4String& itsName, const G4String& /*itsMaterialName*/,
                   G4double itsX1Length, G4double itsY1Length,
                   G4double itsX2Length, G4double itsY2Length,
                   G4double itsZLength,

                   G4double itsXBxLength,
                   G4double itsYBxLength,
                   G4double itsZBxLength,

                   G4double itsXBxPos,
                   G4double itsYBxPos,
                   G4double itsZBxPos,

                   G4bool itsFlagAcceptChildren,
                   G4int depth)
: GateVVolume(itsName, itsFlagAcceptChildren, depth),
  m_trd_solid(0), m_box_solid(0), m_trpd_solid(0), m_trpd_log(0),
  m_Messenger(0)
{

  m_Messenger = new GateTrpdMessenger(this);
  m_trpdLength[0] = itsX1Length;
  m_trpdLength[1] = itsY1Length;

  m_trpdLength[2] = itsX2Length;
  m_trpdLength[3] = itsY2Length;
  m_trpdLength[4] = itsZLength;

  m_trpdLength[5] = itsXBxLength;
  m_trpdLength[6] = itsYBxLength;
  m_trpdLength[7] = itsZBxLength;

  m_trpdLength[8] = itsXBxPos;
  m_trpdLength[9] = itsYBxPos;
  m_trpdLength[10]= itsZBxPos;
}

GateTrpd::~GateTrpd()
{
  if (m_Messenger)
    delete m_Messenger;
}
//------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------
GateTrpd::GateTrpd(const G4String& itsName,
                   G4bool itsFlagAcceptChildren,
                   G4int depth)
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    m_trd_solid(0), m_box_solid(0), m_trpd_solid(0), m_trpd_log(0),
    m_Messenger(0)
{
  m_trpdLength[0] = 1.0*cm;
  m_trpdLength[1] = 1.0*cm;

  m_trpdLength[2] = 1.0*cm;
  m_trpdLength[3] = 1.0*cm;
  m_trpdLength[4] = 1.0*cm;

  m_trpdLength[5] = 1.0*cm;
  m_trpdLength[6] = 1.0*cm;
  m_trpdLength[7] = 1.0*cm;

  m_trpdLength[8] = 1.0*cm;
  m_trpdLength[9] = 1.0*cm;
  m_trpdLength[10]= 1.0*cm;

  m_Messenger = new GateTrpdMessenger(this);
}
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
G4LogicalVolume* GateTrpd::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  G4ThreeVector BoxPos;
  G4RotationMatrix rotMatrix; // Unit matrix created by defaut

  // G4cout << " Trpd : flagUpdateOnly " << flagUpdateOnly << " m_trd_solid =   " << (!m_trpd_solid) << G4endl;

  if (!flagUpdateOnly || !m_trpd_solid) {

    //G4cout << " first "  << G4endl;
    m_trd_solid
      = new G4Trd(GetSolidName(), GetTrpdX1HalfLength(), GetTrpdX2HalfLength(),
                  GetTrpdY1HalfLength(), GetTrpdY2HalfLength(), GetTrpdZHalfLength());

    m_box_solid
      = new G4Box(GetSolidName(),
		  GetTrpdTrudXHalfLength(), GetTrpdTrudYHalfLength(), GetTrpdTrudZHalfLength());

    //! Get here the position of the extruded cylinder
    BoxPos.setX(GetTrpdTrudXPos());
    BoxPos.setY(GetTrpdTrudYPos());
    BoxPos.setZ(GetTrpdTrudZPos());

    //    G4cout << " Create new G4SubtractionSolid  ::: <" << BoxPos << ">" << G4endl;
    delete m_trpd_solid;
    m_trpd_solid
      = new G4SubtractionSolid(GetSolidName(), m_trd_solid, m_box_solid, &rotMatrix, BoxPos);

    delete m_trpd_log;
    m_trpd_log
      = new G4LogicalVolume(m_trpd_solid, mater, GetLogicalVolumeName(),0,0,0);
  }
  else {

    G4cout << " second "  << G4endl;

    BoxPos.setX(GetTrpdTrudXPos());
    BoxPos.setY(GetTrpdTrudYPos());
    BoxPos.setZ(GetTrpdTrudZPos());
    // G4cout << " new val of BoxPos ::: <" << BoxPos << ">" << G4endl;
    // G4Transform3D transform(rotMatrix,BoxPos);

    m_trd_solid->SetXHalfLength1(GetTrpdX1HalfLength());
    m_trd_solid->SetXHalfLength2(GetTrpdX2HalfLength());
    m_trd_solid->SetYHalfLength1(GetTrpdY1HalfLength());
    m_trd_solid->SetYHalfLength2(GetTrpdY2HalfLength());
    m_trd_solid->SetZHalfLength( GetTrpdZHalfLength());

    m_box_solid->SetXHalfLength(GetTrpdTrudXHalfLength());
    m_box_solid->SetYHalfLength(GetTrpdTrudYHalfLength());
    m_box_solid->SetZHalfLength(GetTrpdTrudZHalfLength());

  }
#ifdef debugtrpd
  G4cout << " Returning ...    ::: <" << m_trpd_solid << m_trpd_log << BoxPos << G4endl;
  G4cout << "     local_values ::: <"
	 << " X1: " << m_trpdLength[0]
	 << " Y1: " << m_trpdLength[1]
	 << " X2: " << m_trpdLength[2]
	 << " Y2: " << m_trpdLength[3]
	 << "  Z: " << m_trpdLength[4]
	 << " XLBx: " << m_trpdLength[5]
	 << " YLBx: " << m_trpdLength[6]
	 << " ZLBx: " << m_trpdLength[7]
	 << " XBxPos: " << m_trpdLength[8]
	 << " YBxPos: " << m_trpdLength[9]
	 << " ZBxPos: " << m_trpdLength[10]
	 << G4endl;
#endif
  return m_trpd_log;
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
void GateTrpd::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Shape: extruded trapezoid (trpd) \n";
  G4cout << GateTools::Indent(indent) << "Full Length along X1: " << G4BestUnit(GetTrpdX1Length(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Full Length along Y1: " << G4BestUnit(GetTrpdY1Length(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Full Length along X2: " << G4BestUnit(GetTrpdX2Length(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Full Length along Y2: " << G4BestUnit(GetTrpdY2Length(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Full Length along Z : " << G4BestUnit(GetTrpdZLength(),"Length") << "\n";

  G4cout << GateTools::Indent(indent) << "Extruded box Full Length along X: " << G4BestUnit(GetTrpdTrudXLength(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Extruded box Full Length along Y: " << G4BestUnit(GetTrpdTrudYLength(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Extruded box Full Length along Z: " << G4BestUnit(GetTrpdTrudZLength(),"Length") << "\n";

  G4cout << GateTools::Indent(indent) << "Box center position along X: " << G4BestUnit(GetTrpdTrudXPos()   ,"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Box center position along Y: " << G4BestUnit(GetTrpdTrudYPos()   ,"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Box center position along Z: " << G4BestUnit(GetTrpdTrudZPos()   ,"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Volume TRD - Volume extruded BOX: "        << G4BestUnit(ComputeMyOwnVolume() ,"Volume") << "\n";
}

void GateTrpd::DestroyOwnSolidAndLogicalVolume()
{
  if (m_trpd_log)
    delete m_trpd_log;
  m_trpd_log = 0;

  if (m_trpd_solid)
    delete m_trpd_solid;
  m_trpd_solid = 0;

}
