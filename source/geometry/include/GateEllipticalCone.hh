/*
 * GateEllipticalCone.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATEELLIPTICALCONE_H
#define GATEELLIPTICALCONE_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4ThreeVector.hh"

class G4EllipticalCone;
class GateEllipticalConeMessenger;

class GateEllipticalCone : public GateVVolume
{
public:
  //! Constructor
  GateEllipticalCone(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GateEllipticalCone(const G4String& itsName,
                 const G4String& itsMaterialName,
                 G4double itsXSemiAxis,
                 G4double itsYSemiAxis,
                 G4double itsZLength,
                 G4double itsZCut,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GateEllipticalCone();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateEllipticalCone)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly);

  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param level: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t level);

  inline G4double GetHalfDimension(size_t axis) {
    if (axis==0) {
	return m_xSemiAxis;
    } else if (axis==1) {
	return m_ySemiAxis;
    } else if (axis==2) {
	return m_zLength/2.0;
    }
    return 0.0;
  }

  //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
  //! Returns the volume of the solid
  G4double ComputeMyOwnVolume()  const;

  //! \name getters and setters
  //@{
  inline G4double GetEllipticalConeXSemiAxis() 	{ return m_xSemiAxis; }
  inline G4double GetEllipticalConeYSemiAxis() 	{ return m_ySemiAxis; }
  inline G4double GetEllipticalConeZLength() 	{ return m_zLength; }
  inline G4double GetEllipticalConeZCut() 	{ return m_zCut; }

  void SetEllipticalConeXSemiAxis(G4double val)	{ m_xSemiAxis = val; }
  void SetEllipticalConeYSemiAxis(G4double val)	{ m_ySemiAxis = val; }
  void SetEllipticalConeZLength(G4double val)	{ m_zLength = val; }
  void SetEllipticalConeZCut(G4double val)	{ m_zCut = val; }
  //@}


private:
  //! \name own geometry
  //@{
  G4EllipticalCone*          m_ellipticalcone_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_ellipticalcone_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4double m_xSemiAxis;		//!< x semiaxis
  G4double m_ySemiAxis;		//!< y semiaxis
  G4double m_zLength;		//!< z height
  G4double m_zCut;		//!< z cut
  //@}

  //! Messenger
  GateEllipticalConeMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(ellipticalcone,GateEllipticalCone)


#endif /* GATEELLIPTICALCONE_H */
