/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateGeneralTrpd.hh"
#include "GateGeneralTrpdMessenger.hh"

#include "G4Trap.hh"
#include "G4LogicalVolume.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"
#include "G4PVPlacement.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "G4Transform3D.hh"
#include "GateTools.hh"
#include "G4Material.hh"

//---------------------------------------------------------------------------------------------------
GateGeneralTrpd::GateGeneralTrpd(const G4String& itsName, const G4String& itsMaterialName,
                                 G4double itsX1Length, G4double itsY1Length,
                                 G4double itsX2Length, G4double itsY2Length,
                                 G4double itsX3Length, G4double itsX4Length,
                                 G4double itsZLength,
                                 G4double itsTheta,    G4double itsPhi,
                                 G4double itsAlp1,     G4double itsAlp2,
                                 G4bool itsFlagAcceptChildren,
                                 G4int depth)
  : GateVVolume(itsName,itsFlagAcceptChildren, depth),
    m_general_trpd_solid(0), m_general_trpd_log(0),
    m_Messenger(0)
{
  m_Messenger = new GateGeneralTrpdMessenger(this);
  m_X1= itsX1Length;
  m_X2= itsX2Length;
  m_X3= itsX3Length;
  m_X4= itsX4Length;
  m_Y1= itsY1Length;
  m_Y2= itsY2Length;
  m_Z= itsZLength;
  m_Theta= itsTheta;
  m_Phi= itsPhi;
  m_Alp1= itsAlp1;
  m_Alp2= itsAlp2;
  
  SetMaterialName(itsMaterialName);
}
//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
GateGeneralTrpd::GateGeneralTrpd(const G4String& itsName,
                                 G4bool itsFlagAcceptChildren,
                                 G4int depth)
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    m_general_trpd_solid(0), m_general_trpd_log(0),pGeneralTrpdPhys(0),
    m_Messenger(0)
{
  SetMaterialName("Vacuum");
  m_Messenger = new GateGeneralTrpdMessenger(this);
  m_X1= 0.;
  m_X2= 0.;
  m_X3= 0.;
  m_X4= 0.;
  m_Y1= 0.;
  m_Y2= 0.;
  m_Z= 0.;
  m_Theta= 0.;
  m_Phi= 0.;
  m_Alp1= 0.;
  m_Alp2= 0.;  
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
GateGeneralTrpd::~GateGeneralTrpd()
{  
  if (m_Messenger) delete m_Messenger;
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
G4LogicalVolume* GateGeneralTrpd::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  G4RotationMatrix rotMatrix; // Unit matrix created by defaut 
  if (!flagUpdateOnly|| !m_general_trpd_solid) {
    m_general_trpd_solid
      = new G4Trap (GetSolidName(), 
                    GetTrpdZLength()/2.0, 
                    GetTrpdTheta(), 
                    GetTrpdPhi(), 
                    GetTrpdY1Length()/2.0, 
                    GetTrpdX1Length()/2.0, 
                    GetTrpdX2Length()/2.0, 
                    GetTrpdAlp1(), 
                    GetTrpdY2Length()/2.0, 
                    GetTrpdX3Length()/2.0, 
                    GetTrpdX4Length()/2.0, 
                    GetTrpdAlp2());
    delete m_general_trpd_log;
    m_general_trpd_log
      = new G4LogicalVolume(m_general_trpd_solid, mater, GetLogicalVolumeName(),0,0,0);
  }
  else {
    // DS : I comment the following lines because it creates bug when moving/repeating a volume like this
    /*
      m_general_trpd_solid->SetAllParameters (
      GetTrpdZLength(),
      GetTrpdTheta(),
      GetTrpdPhi(),
      GetTrpdY1Length(),
      GetTrpdX1Length(),
      GetTrpdX2Length(),
      GetTrpdAlp1(),
      GetTrpdX2Length(),
      GetTrpdX3Length(),
      GetTrpdX4Length(),
      GetTrpdAlp2() );
    */

  }
  return m_general_trpd_log;
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
void GateGeneralTrpd::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Shape: extruded trapezoid (trpd) \n";
  G4cout << GateTools::Indent(indent) << "Full Length along X1: " << G4BestUnit(GetTrpdX1Length(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Full Length along Y1: " << G4BestUnit(GetTrpdY1Length(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Full Length along X2: " << G4BestUnit(GetTrpdX2Length(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Full Length along Y2: " << G4BestUnit(GetTrpdY2Length(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Full Length along Z : " << G4BestUnit(GetTrpdZLength(),"Length") << "\n";
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
void GateGeneralTrpd::DestroyOwnVolume()
{
  if (m_general_trpd_log) delete m_general_trpd_log;
  m_general_trpd_log = 0; 

  if (m_general_trpd_solid) delete m_general_trpd_solid;
  m_general_trpd_solid = 0;
}
//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
void GateGeneralTrpd::DestroyOwnSolidAndLogicalVolume()
{
  if (m_general_trpd_log)   delete m_general_trpd_log;
  if (m_general_trpd_solid) delete m_general_trpd_solid;
  m_general_trpd_solid = 0;
}  
//------------------------------------------------------------------------------------------------
