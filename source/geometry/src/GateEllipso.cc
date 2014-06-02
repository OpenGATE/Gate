/*-----------------------------------
-----------------------------------*/

#include "GateEllipso.hh"  
#include "GateEllipsoMessenger.hh" 

#include "GateTools.hh"

#include "G4Ellipsoid.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Colour.hh"
#include "G4UnitsTable.hh"

#include "G4Material.hh"

#include "G4ThreeVector.hh"

#include "globals.hh"


//----------------------------------------------------------------------------------
GateEllipso::GateEllipso(const G4String& itsName, const G4String& /*itsMaterialName*/, // commented also in GateCone
			 G4double itspxSemiAxis, G4double itspySemiAxis, G4double itspzSemiAxis,
			 G4double itspzBottomCut, G4double itspzTopCut, 
			 G4bool itsFlagAcceptChildren, G4int depth)
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    m_ellipsoid_solid(0), m_ellipsoid_log(0),
    m_ellipsopxSemiAxis(itspxSemiAxis), 
    m_ellipsopySemiAxis(itspySemiAxis),
    m_ellipsopzSemiAxis(itspzSemiAxis),
    m_ellipsopzBottomCut(itspzBottomCut),
    m_ellipsopzTopCut(itspzTopCut),
    m_Messenger(0)
{
  m_Messenger = new GateEllipsoMessenger(this);
}


//----------------------------------------------------------------------------------
GateEllipso::GateEllipso(const G4String& itsName, 
			 G4bool itsFlagAcceptChildren, 
			 G4int depth)
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    m_ellipsoid_solid(0), m_ellipsoid_log(0), m_Messenger(0)
{
  m_ellipsopxSemiAxis = 1.0*cm ;
  m_ellipsopySemiAxis = 1.0*cm ;
  m_ellipsopzSemiAxis = 1.0*cm ;
  m_ellipsopzBottomCut =0;
  m_ellipsopzTopCut=0;

  m_Messenger = new GateEllipsoMessenger(this);
}


//----------------------------------------------------------------------------------
GateEllipso::~GateEllipso()
{
  delete m_Messenger;
}


//----------------------------------------------------------------------------------
G4LogicalVolume* GateEllipso::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  
  if(!flagUpdateOnly || !m_ellipsoid_solid)
    {
      m_ellipsoid_solid = new G4Ellipsoid(GetSolidName(), 
					  m_ellipsopxSemiAxis,
					  m_ellipsopySemiAxis,
					  m_ellipsopzSemiAxis,
					  m_ellipsopzBottomCut,
					  m_ellipsopzTopCut);
      m_ellipsoid_log = new G4LogicalVolume(m_ellipsoid_solid, mater, GetLogicalVolumeName());
    }
  else
    {
      G4cout << " second " << G4endl;
      m_ellipsoid_solid->SetSemiAxis(GetEllipsopxSemiAxis(), GetEllipsopySemiAxis(), GetEllipsopzSemiAxis());
      m_ellipsoid_solid->SetZCuts(GetEllipsopzBottomCut(),GetEllipsopzTopCut());

    }
  return m_ellipsoid_log;
}


//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------

void GateEllipso::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Shape:          Ellipsoid \n";
  G4cout << GateTools::Indent(indent) << "Half x:         " << G4BestUnit(m_ellipsopxSemiAxis, "Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Half y:         " << G4BestUnit(m_ellipsopySemiAxis, "Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Half z:         " << G4BestUnit(m_ellipsopzSemiAxis, "Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Bottom cut z:   " << G4BestUnit(m_ellipsopzBottomCut, "Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Top cut z:      " << G4BestUnit(m_ellipsopzTopCut, "Length") << "\n";

}

//----------------------------------------------------------------------------------
void GateEllipso::DestroyOwnSolidAndLogicalVolume()
{
  if(m_ellipsoid_log)
    {
      delete m_ellipsoid_log;
    }
  m_ellipsoid_log = 0;

  if(m_ellipsoid_solid)
    {
      delete m_ellipsoid_solid;
    }
  m_ellipsoid_solid = 0;
}

//----------------------------------------------------------------------------------
