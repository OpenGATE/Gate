/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTrap.hh"
#include "GateTrapMessenger.hh"

#include "G4Trap.hh"
#include "G4LogicalVolume.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"

class G4Material;

GateTrap::GateTrap(const G4String& itsName, const G4String& /*itsMaterialName*/,
			 G4double itsDz,G4double itsTheta, G4double itsPhi,
			 G4double itsDy1,G4double itsDx1,G4double itsDx2,G4double itsAlp1,
			 G4double itsDy2,G4double itsDx3,G4double itsDx4,G4double itsAlp2)
  : GateVVolume(itsName, true, 0),
    m_TrapDz(itsDz),m_TrapTheta(itsTheta),m_TrapPhi(itsPhi),m_TrapDy1(itsDy1),
    m_TrapDx1(itsDx1),m_TrapDx2(itsDx2),m_TrapAlp1(itsAlp1),m_TrapDy2(itsDy2),
    m_TrapDx3(itsDx3),m_TrapDx4(itsDx4),m_TrapAlp2(itsAlp2),
    m_Trap_solid(0), m_Trap_log(0), m_Messenger(0)
{
    m_Messenger = new GateTrapMessenger(this);
}


GateTrap::GateTrap(const G4String& itsName,
		   G4bool itsFlagAcceptChildren,
	           G4int depth)
: GateVVolume(itsName, itsFlagAcceptChildren, depth),
  m_Trap_solid(0), m_Trap_log(0), m_Messenger(0) 
{
    m_TrapDz    = 1.0 * cm;
    m_TrapTheta = 90.0 * deg;
    m_TrapPhi   = 180.0 * deg;
    m_TrapDy1   = 1.0 * cm;
    m_TrapDx1   = 1.0 * cm;
    m_TrapDx2   = 1.0 * cm;
    m_TrapAlp1  = 30.0*deg;
    m_TrapDy2   = 1.0 * cm;
    m_TrapDx3   = 1.0 * cm;
    m_TrapDx4   = 1.0 * cm;
    m_TrapAlp2  = 30.0*deg;
     
  
     m_Messenger = new GateTrapMessenger(this);
}



GateTrap::~GateTrap()
{
  delete m_Messenger;
}

G4LogicalVolume* GateTrap::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly)
  {
    G4cout << " in the if condition " << " x1 = " << GetTrapDx1() << "  x2 = " << GetTrapDx2() << "   x3 = " << GetTrapDx3() << "  x4 = "  << GetTrapDx4() <<  G4endl;

    m_Trap_solid = new G4Trap(GetSolidName(), GetTrapDz(), GetTrapTheta(), GetTrapPhi(), GetTrapDy1(), GetTrapDx1(),
      		   GetTrapDx2(), GetTrapAlp1(), GetTrapDy2(), GetTrapDx3(), GetTrapDx4(), GetTrapAlp2());


    m_Trap_log   = new G4LogicalVolume(m_Trap_solid, mater, GetLogicalVolumeName(), 0, 0, 0);
  }

  return m_Trap_log;
}

void GateTrap::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Shape:                                                 Trap\n";
  G4cout << GateTools::Indent(indent) << "Half-length z-axis:                                   " << G4BestUnit(GetTrapDz(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Polar angle:                                          " << G4BestUnit(GetTrapTheta(),"Angle") << "\n";
  G4cout << GateTools::Indent(indent) << "Azimuthal angle:                                      " << G4BestUnit(GetTrapPhi(),"Angle") << "\n";
  G4cout << GateTools::Indent(indent) << "Half-length y-axis face at -pDz:                      " << G4BestUnit(GetTrapDy1(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Half-length x-axis side at -pDy1 of the face at -pDz: " << G4BestUnit(GetTrapDx1(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Half-length x-axis side at +pDy1 of the face at -pDz: " << G4BestUnit(GetTrapDx2(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Angle with y-axis face at -pDz:                       " << G4BestUnit(GetTrapAlp1(),"Angle") << "\n";
  G4cout << GateTools::Indent(indent) << "Half-length y-axis face at +pDz:                      " << G4BestUnit(GetTrapDy2(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Half-length x-axis side at -pDy1 of the face at +pDz: " << G4BestUnit(GetTrapDx3(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Half-length x-axis side at +pDy1 of the face at +pDz: " << G4BestUnit(GetTrapDx4(),"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Angle with y-axis face at +pDz:                       " << G4BestUnit(GetTrapAlp2(),"Angle") << "\n";

}

void GateTrap::DestroyOwnSolidAndLogicalVolume()
{
  if (m_Trap_log)
    delete m_Trap_log;
  m_Trap_log = 0;

  if (m_Trap_solid)
    delete m_Trap_solid;
  m_Trap_solid = 0;

}

