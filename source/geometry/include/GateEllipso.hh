/*-----------------------------------
-----------------------------------*/


#ifndef GateEllipso_h
#define GateEllipso_h 1

#include "globals.hh"

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"

#include "G4Ellipsoid.hh"

class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Material;

class GateEllipsoMessenger;

class GateEllipso : public GateVVolume
{
public:

  GateEllipso(const G4String& itsName,
	      G4bool acceptsChildren=true,
	      G4int depth=0);

  // constructor
  GateEllipso(const G4String& itsName, const G4String& itsMaterialName,
	      G4double itspxSemiAxis, G4double itspySemiAxis, G4double itspzSemiAxis,
	      G4double itspzBottomCut,  G4double itspzTopCut,
	      G4bool itsFlagAcceptChildren=true, G4int depth=0);

  // destructor
  virtual ~GateEllipso();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateEllipso)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool);
  virtual void DestroyOwnSolidAndLogicalVolume();
  virtual void DescribeMyself(size_t indent);

  inline G4double GetHalfDimension(size_t axis)
  {
    if(axis==0){return GetEllipsopxSemiAxis();}
    else if(axis==1){return GetEllipsopySemiAxis();}
    else if(axis==2){return GetEllipsopzSemiAxis();}
    else {
      GateError("Error axis must be 0,1 or 2");
      return 0;
    }
  }


  // name getters and setters
  inline G4double GetEllipsopxSemiAxis() {return m_ellipsopxSemiAxis;};
  inline G4double GetEllipsopySemiAxis() {return m_ellipsopySemiAxis;};
  inline G4double GetEllipsopzSemiAxis() {return m_ellipsopzSemiAxis;};
  inline G4double GetEllipsopzBottomCut() {return m_ellipsopzBottomCut;};
  inline G4double GetEllipsopzTopCut() {return m_ellipsopzTopCut;};
  inline G4double GetEllipsopzTotalHeight() {return m_ellipsopzTopCut-m_ellipsopzBottomCut;};

  void SetEllipsopxSemiAxis(G4double val)
  {
    m_ellipsopxSemiAxis = val;
  }
  void SetEllipsopySemiAxis(G4double val)
  {
    m_ellipsopySemiAxis = val;
  }
  void SetEllipsopzSemiAxis(G4double val)
  {
    m_ellipsopzSemiAxis = val;
  }
  void SetEllipsopzBottomCut(G4double val)
  {
    m_ellipsopzBottomCut = val;
  }
  void SetEllipsopzTopCut(G4double val)
  {
    m_ellipsopzTopCut = val;
  }


private:
  G4Ellipsoid* m_ellipsoid_solid;
  G4LogicalVolume* m_ellipsoid_log;

  G4double m_ellipsopxSemiAxis;
  G4double m_ellipsopySemiAxis;
  G4double m_ellipsopzSemiAxis;
  G4double m_ellipsopzBottomCut;
  G4double m_ellipsopzTopCut;


  GateEllipsoMessenger* m_Messenger;


};

MAKE_AUTO_CREATOR_VOLUME(ellipsoid, GateEllipso)

#endif
