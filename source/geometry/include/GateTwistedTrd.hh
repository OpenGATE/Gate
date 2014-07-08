/*
 * GateTwistedTrd.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETWISTEDTRD_H
#define GATETWISTEDTRD_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4ThreeVector.hh"

class G4TwistedTrd;
class GateTwistedTrdMessenger;

class GateTwistedTrd : public GateVVolume
{
public:
  //! Constructor
  GateTwistedTrd(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GateTwistedTrd(const G4String& itsName,
                 const G4String& itsMaterialName,
                 G4double itsX1Length,
                 G4double itsX2Length,
                 G4double itsY1Length,
                 G4double itsY2Length,
                 G4double itsZLength,
                 G4double itsTwistAngle,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GateTwistedTrd();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateTwistedTrd)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly);

  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param indent: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t level);

  inline G4double GetHalfDimension(size_t axis) {
    if (axis==0) {
    	return 0.25*(m_X1Length + m_X2Length);
    } else if (axis==1) {
	return 0.25*(m_Y1Length + m_Y2Length);
    } else if (axis==2) {
	return 0.5*m_ZLength;
    }
    return 0.0;
  }

  //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
  //! Returns the volume of the solid
  G4double ComputeMyOwnVolume()  const;

  //! \name getters and setters
  //@{
  inline G4double GetTwistedTrdX1Length() 	{ return m_X1Length; }
  inline G4double GetTwistedTrdX2Length() 	{ return m_X2Length; }
  inline G4double GetTwistedTrdY1Length() 	{ return m_Y1Length; }
  inline G4double GetTwistedTrdY2Length() 	{ return m_Y2Length; }
  inline G4double GetTwistedTrdZLength() 	{ return m_ZLength; }
  inline G4double GetTwistedTrdTwistAngle() 	{ return m_twistAngle; }

  void SetTwistedTrdX1Length(G4double val)	{ m_X1Length = val; }
  void SetTwistedTrdX2Length(G4double val)	{ m_X2Length = val; }
  void SetTwistedTrdY1Length(G4double val)	{ m_Y1Length = val; }
  void SetTwistedTrdY2Length(G4double val)	{ m_Y2Length = val; }
  void SetTwistedTrdZLength(G4double val)	{ m_ZLength = val; }
  void SetTwistedTrdTwistAngle(G4double val)	{ m_twistAngle = val; }
  //@}


private:
  //! \name own geometry
  //@{
  G4TwistedTrd*          m_twistedtrd_solid;       	    //!< solid pointer
  G4LogicalVolume*   m_twistedtrd_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4double m_X1Length;
  G4double m_X2Length;
  G4double m_Y1Length;
  G4double m_Y2Length;
  G4double m_ZLength;
  G4double m_twistAngle;	//!< twist angle
  //@}

  //! Messenger
  GateTwistedTrdMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(twistedtrd,GateTwistedTrd)


#endif /* GATETWISTEDTRD_H */
