/*
 * GateTwistedBox.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETWISTEDBOX_H
#define GATETWISTEDBOX_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4ThreeVector.hh"

class G4TwistedBox;
class GateTwistedBoxMessenger;

class GateTwistedBox : public GateVVolume
{
public:
  //! Constructor
  GateTwistedBox(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GateTwistedBox(const G4String& itsName,
                 const G4String& itsMaterialName,
                 G4double itsXLength,
                 G4double itsYLength,
                 G4double itsZLength,
                 G4double itsTwistAngle,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GateTwistedBox();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateTwistedBox)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly);

  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param indent: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t level);

  inline G4double GetHalfDimension(size_t axis) {
    return m_Length[axis]/2.0;
  }

  //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
  //! Returns the volume of the solid
  G4double ComputeMyOwnVolume()  const;

  //! \name getters and setters
  //@{
  inline G4double GetTwistedBoxXLength() 	{ return m_Length.x(); }
  inline G4double GetTwistedBoxYLength() 	{ return m_Length.y(); }
  inline G4double GetTwistedBoxZLength() 	{ return m_Length.z(); }
  inline G4double GetTwistedBoxTwistAngle() 	{ return m_twistAngle; }

  void SetTwistedBoxXLength(G4double val)	{ m_Length.setX(val); }
  void SetTwistedBoxYLength(G4double val)	{ m_Length.setY(val); }
  void SetTwistedBoxZLength(G4double val)	{ m_Length.setZ(val); }
  void SetTwistedBoxTwistAngle(G4double val)	{ m_twistAngle = val; }

  //@}


private:
  //! \name own geometry
  //@{
  G4TwistedBox*          m_twistedbox_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_twistedbox_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4ThreeVector m_Length;		//!< x, y, z length
  G4double m_twistAngle;	//!< twist angle
  //@}

  //! Messenger
  GateTwistedBoxMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(twistedbox,GateTwistedBox)


#endif /* GATETWISTEDBOX_H */
