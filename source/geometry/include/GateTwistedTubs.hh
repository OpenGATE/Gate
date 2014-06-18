/*
 * GateTwistedTubs.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETWISTEDTUBS_H
#define GATETWISTEDTUBS_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4ThreeVector.hh"

class G4TwistedTubs;
class GateTwistedTubsMessenger;

class GateTwistedTubs : public GateVVolume
{
public:
  //! Constructor
  GateTwistedTubs(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GateTwistedTubs(const G4String& itsName,
                 const G4String& itsMaterialName,
                 G4double itsTwistAngle,
                 G4double itsInnerR,
                 G4double itsOuterR,
                 G4double itsNegZ,
                 G4double itsPosZ,
                 G4int itsNSegment,
                 G4double itsTotalPhi,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GateTwistedTubs();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateTwistedTubs)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly);

  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param level: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t level);

  inline G4double GetHalfDimension(size_t axis) {
    if (axis==0) {
	return m_outerR;
    } else if (axis==1) {
	return m_outerR;
    } else if (axis==2) {
	return (m_posZ + m_negZ)/2.0;
    }
    return 0.0;
  }

  //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
  //! Returns the volume of the solid
  G4double ComputeMyOwnVolume()  const;

  //! \name getters and setters
  //@{
  inline G4double GetTwistedTubsInnerR() 	{ return m_innerR; }
  inline G4double GetTwistedTubsOuterR() 	{ return m_outerR; }
  inline G4double GetTwistedTubsPosZ() 		{ return m_posZ; }
  inline G4double GetTwistedTubsNegZ() 		{ return m_negZ; }
  inline G4double GetTwistedTubsTotalPhi() 	{ return m_totalPhi; }
  inline G4double GetTwistedTubsTwistAngle() 	{ return m_twistAngle; }
  inline G4int GetTwistedTubsNSegment() 	{ return m_nSegment; }

  void SetTwistedTubsInnerR(G4double val)	{ m_innerR = val; }
  void SetTwistedTubsOuterR(G4double val)	{ m_outerR = val; }
  void SetTwistedTubsPosZ(G4double val)		{ m_posZ = val; }
  void SetTwistedTubsNegZ(G4double val)		{ m_negZ = val; }
  void SetTwistedTubsTotalPhi(G4double val)	{ m_totalPhi = val; }
  void SetTwistedTubsTwistAngle(G4double val)	{ m_twistAngle = val; }
  void SetTwistedTubsNSegment(G4int val)	{ m_nSegment = val; }

  //@}


private:
  //! \name own geometry
  //@{
  G4TwistedTubs*          m_twistedtubs_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_twistedtubs_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4double m_twistAngle;	//!< twist angle
  G4double m_innerR;		//!< Inner radius at z=0
  G4double m_outerR;		//!< Outer radius at z=0
  G4double m_negZ;		//!< z coordinate of - endplate
  G4double m_posZ;		//!< z coordinate of + endplate
  G4int m_nSegment;        	//!< Number of segments in totalPhi
  G4double m_totalPhi;       	//!< Total angle of all segments

  //@}

  //! Messenger
  GateTwistedTubsMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(twistedtubesegment,GateTwistedTubs)


#endif /* GATETWISTEDTUBS_H */
