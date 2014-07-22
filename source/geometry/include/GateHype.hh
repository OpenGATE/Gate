/*
 * GateHype.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATEHYPE_H
#define GATEHYPE_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4ThreeVector.hh"

class G4Hype;
class GateHypeMessenger;

class GateHype : public GateVVolume
{
public:
  //! Constructor
  GateHype(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GateHype(const G4String& itsName,
                 const G4String& itsMaterialName,
                 G4double itsInnerR,
                 G4double itsOuterR,
                 G4double itsInnerStereo,
                 G4double itsOuterStereo,
                 G4double itsZLength,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GateHype();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateHype)

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
	return m_zLength/2.0;
    }
    return 0.0;
  }

  //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
  //! Returns the volume of the solid
  G4double ComputeMyOwnVolume()  const;

  //! \name getters and setters
  //@{
  inline G4double GetHypeInnerR() 	{ return m_innerR; }
  inline G4double GetHypeOuterR() 	{ return m_outerR; }
  inline G4double GetHypeInnerStereo() 	{ return m_innerStereo; }
  inline G4double GetHypeOuterStereo() 	{ return m_outerStereo; }
  inline G4double GetHypeZLength() 	{ return m_zLength; }

  void SetHypeInnerR(G4double val)	{ m_innerR = val; }
  void SetHypeOuterR(G4double val)	{ m_outerR = val; }
  void SetHypeInnerStereo(G4double val)	{ m_innerStereo = val; }
  void SetHypeOuterStereo(G4double val)	{ m_outerStereo = val; }
  void SetHypeZLength(G4double val)	{ m_zLength = val; }
  //@}


private:
  //! \name own geometry
  //@{
  G4Hype*          m_hype_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_hype_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4double m_innerR;		//!< Inner radius at z=0
  G4double m_outerR;		//!< Outer radius at z=0
  G4double m_innerStereo;	//!< inner stereo angle
  G4double m_outerStereo;	//!< outer stereo angle
  G4double m_zLength;		//!< z length
  //@}

  //! Messenger
  GateHypeMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(hypertube,GateHype)


#endif /* GATEHYPE_H */
