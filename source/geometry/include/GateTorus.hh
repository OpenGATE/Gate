/*
 * GateTorus.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETORUS_H
#define GATETORUS_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4ThreeVector.hh"

class G4Torus;
class GateTorusMessenger;

class GateTorus : public GateVVolume
{
public:
  //! Constructor
  GateTorus(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GateTorus(const G4String& itsName,
                 const G4String& itsMaterialName,
                 G4double itsInnerR,
                 G4double itsOuterR,
                 G4double itsStartPhi,
                 G4double itsDeltaPhi,
                 G4double itsTorusR,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GateTorus();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateTorus)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly);

  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param level: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t level);

  inline G4double GetHalfDimension(size_t axis) {
    if (axis==0) {
	return m_torusR;
    } else if (axis==1) {
	return m_torusR;
    } else if (axis==2) {
	return m_outerR;
    }
    return 0.0;
  }

  //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
  //! Returns the volume of the solid
  G4double ComputeMyOwnVolume()  const;

  //! \name getters and setters
  //@{
  inline G4double GetTorusInnerR() 	{ return m_innerR; }
  inline G4double GetTorusOuterR() 	{ return m_outerR; }
  inline G4double GetTorusStartPhi() 	{ return m_startPhi; }
  inline G4double GetTorusDeltaPhi() 	{ return m_deltaPhi; }
  inline G4double GetTorusTorusR() 	{ return m_torusR; }

  void SetTorusInnerR(G4double val)	{ m_innerR = val; }
  void SetTorusOuterR(G4double val)	{ m_outerR = val; }
  void SetTorusStartPhi(G4double val)	{ m_startPhi = val; }
  void SetTorusDeltaPhi(G4double val)	{ m_deltaPhi = val; }
  void SetTorusTorusR(G4double val)	{ m_torusR = val; }
  //@}


private:
  //! \name own geometry
  //@{
  G4Torus*          m_torus_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_torus_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4double m_innerR;		//!< Inner radius
  G4double m_outerR;		//!< Outer radius
  G4double m_startPhi;		//!< starting angle of the torus segment
  G4double m_deltaPhi;		//!< delta angle of the segment
  G4double m_torusR;		//!< Torus radius
  //@}

  //! Messenger
  GateTorusMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(torus,GateTorus)


#endif /* GATETORUS_H */
