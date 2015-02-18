/*
 * GateTwistedTrap.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETWISTEDTRAP_H
#define GATETWISTEDTRAP_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4ThreeVector.hh"

class G4TwistedTrap;
class GateTwistedTrapMessenger;

class GateTwistedTrap : public GateVVolume
{
public:
  //! Constructor
  GateTwistedTrap(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GateTwistedTrap(const G4String& itsName,
                 const G4String& itsMaterialName,
                 G4double itsZLength,
                 G4double itsYMinusLength,
                 G4double itsYPlusLength,
                 G4double itsX1MinusLength,
                 G4double itsX2MinusLength,
                 G4double itsX1PlusLength,
                 G4double itsX2PlusLength,
                 G4double itsTwistAngle,
                 G4double itsPolarAngle,
                 G4double itsAzimuthalAngle,
                 G4double itsTiltAngle,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GateTwistedTrap();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateTwistedTrap)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly);

  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param level: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t level);

  inline G4double GetHalfDimension(size_t axis) {
    if (axis==0) {
	return (m_x1MinusLength+m_x1PlusLength+m_x2MinusLength+m_x2PlusLength)/4.0;
    } else if (axis==1) {
	return (m_yMinusLength + m_yPlusLength)/4.0;
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
  inline G4double GetTwistedTrapZLength() 	{ return m_zLength; }
  inline G4double GetTwistedTrapYMinusLength() 	{ return m_yMinusLength; }
  inline G4double GetTwistedTrapYPlusLength() 	{ return m_yPlusLength; }
  inline G4double GetTwistedTrapX1MinusLength() { return m_x1MinusLength; }
  inline G4double GetTwistedTrapX2MinusLength() { return m_x2MinusLength; }
  inline G4double GetTwistedTrapX1PlusLength() 	{ return m_x1PlusLength; }
  inline G4double GetTwistedTrapX2PlusLength() 	{ return m_x2PlusLength; }
  inline G4double GetTwistedTrapTwistAngle() 	{ return m_twistAngle; }
  inline G4double GetTwistedTrapPolarAngle() 	{ return m_polarAngle; }
  inline G4double GetTwistedTrapAzimuthalAngle() { return m_azimuthalAngle; }
  inline G4double GetTwistedTrapTiltAngle() 	{ return m_tiltAngle; }

  void SetTwistedTrapZLength(G4double val) 		{ m_zLength = val; }
  void SetTwistedTrapYMinusLength(G4double val) 	{ m_yMinusLength = val; }
  void SetTwistedTrapYPlusLength(G4double val) 		{ m_yPlusLength = val; }
  void SetTwistedTrapX1MinusLength(G4double val) 	{ m_x1MinusLength = val; }
  void SetTwistedTrapX2MinusLength(G4double val) 	{ m_x2MinusLength = val; }
  void SetTwistedTrapX1PlusLength(G4double val) 	{ m_x1PlusLength = val; }
  void SetTwistedTrapX2PlusLength(G4double val) 	{ m_x2PlusLength = val; }
  void SetTwistedTrapTwistAngle(G4double val) 		{ m_twistAngle = val; }
  void SetTwistedTrapPolarAngle(G4double val) 		{ m_polarAngle = val; }
  void SetTwistedTrapAzimuthalAngle(G4double val) 	{ m_azimuthalAngle = val; }
  void SetTwistedTrapTiltAngle(G4double val) 		{ m_tiltAngle = val; }
  //@}

private:
  //! \name own geometry
  //@{
  G4TwistedTrap*     m_twistedtrap_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_twistedtrap_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4double m_zLength; 		//!< z length
  G4double m_yMinusLength; 	//!< y length at -z/2
  G4double m_yPlusLength; 	//!< y length at +z/2
  G4double m_x1MinusLength; 	//!< x1 length at -z/2, -y/2
  G4double m_x2MinusLength; 	//!< x2 length at -z/2, +y/2
  G4double m_x1PlusLength; 	//!< x1 length at +z/2, -y/2
  G4double m_x2PlusLength; 	//!< x2 length at +z/2, +y/2

  G4double m_twistAngle;	//!< twist angle
  G4double m_polarAngle;	//!< polar angle
  G4double m_azimuthalAngle;	//!< azimuthal angle
  G4double m_tiltAngle;		//!< tilt angle
  //@}

  //! Messenger
  GateTwistedTrapMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(twistedtrap,GateTwistedTrap)


#endif /* GATETWISTEDTRAP_H */
