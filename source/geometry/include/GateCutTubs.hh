/*
 * GateCutTubs.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATECUTTUBS_H
#define GATECUTTUBS_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4ThreeVector.hh"

class G4CutTubs;
class GateCutTubsMessenger;

class GateCutTubs : public GateVVolume
{
public:
  //! Constructor
  GateCutTubs(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GateCutTubs(const G4String& itsName,
                 const G4String& itsMaterialName,
                 G4double itsInnerR,
                 G4double itsOuterR,
                 G4double itsStartPhi,
                 G4double itsDeltaPhi,
                 G4double itsZLength,
                 G4ThreeVector itsNegNorm,
                 G4ThreeVector itsPosNorm,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GateCutTubs();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateCutTubs)

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
  inline G4double GetCutTubsInnerR() 	{ return m_innerR; }
  inline G4double GetCutTubsOuterR() 	{ return m_outerR; }
  inline G4double GetCutTubsStartPhi() 	{ return m_startPhi; }
  inline G4double GetCutTubsDeltaPhi() 	{ return m_deltaPhi; }
  inline G4double GetCutTubsZLength() 	{ return m_zLength; }
  inline G4ThreeVector GetCutTubsNegNorm() { return m_negNorm; }
  inline G4ThreeVector GetCutTubsPosNorm() { return m_posNorm; }

  void SetCutTubsInnerR(G4double val)	{ m_innerR = val; }
  void SetCutTubsOuterR(G4double val)	{ m_outerR = val; }
  void SetCutTubsStartPhi(G4double val)	{ m_startPhi = val; }
  void SetCutTubsDeltaPhi(G4double val)	{ m_deltaPhi = val; }
  void SetCutTubsZLength(G4double val)	{ m_zLength = val; }
  void SetCutTubsNegNorm(G4ThreeVector val)	{ m_negNorm = val; }
  void SetCutTubsPosNorm(G4ThreeVector val)	{ m_posNorm = val; }
  //@}


private:
  //! \name own geometry
  //@{
  G4CutTubs*          m_cuttubs_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_cuttubs_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4double m_innerR;		//!< Inner radius
  G4double m_outerR;		//!< Outer radius
  G4double m_startPhi;		//!< starting angle of the tube segment
  G4double m_deltaPhi;		//!< delta angle of the tube segment
  G4double m_zLength;		//!< z length of the tube
  G4ThreeVector m_negNorm;	//!< outside normal of the cutting plane at -z/2
  G4ThreeVector m_posNorm;	//!< outside normal of the cutting plane at +z/2

  //@}

  //! Messenger
  GateCutTubsMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(cuttube,GateCutTubs)


#endif /* GATECUTTUBS_H */
