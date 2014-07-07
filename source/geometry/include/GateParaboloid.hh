/*
 * GateParaboloid.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATEPARABOLOID_H
#define GATEPARABOLOID_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4ThreeVector.hh"

class G4Paraboloid;
class GateParaboloidMessenger;

class GateParaboloid : public GateVVolume
{
public:
  //! Constructor
  GateParaboloid(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GateParaboloid(const G4String& itsName,
                 const G4String& itsMaterialName,
                 G4double itsNegR,
                 G4double itsPosR,
                 G4double itsZLength,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GateParaboloid();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateParaboloid)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly);

  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param level: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t level);

  inline G4double GetHalfDimension(size_t axis) {
    if (axis==0) {
	return m_posR;
    } else if (axis==1) {
	return m_posR;
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
  inline G4double GetParaboloidNegativeR() 	{ return m_negR; }
  inline G4double GetParaboloidPositiveR() 	{ return m_posR; }
  inline G4double GetParaboloidZLength() 	{ return m_zLength; }

  void SetParaboloidNegativeR(G4double val)	{ m_negR = val; }
  void SetParaboloidPositiveR(G4double val)	{ m_posR = val; }
  void SetParaboloidZLength(G4double val)	{ m_zLength = val; }
  //@}


private:
  //! \name own geometry
  //@{
  G4Paraboloid*          m_paraboloid_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_paraboloid_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4double m_negR;		//!< radius at -z/2
  G4double m_posR;		//!< radius at +z/2
  G4double m_zLength;		//!< z length
  //@}

  //! Messenger
  GateParaboloidMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(paraboloid,GateParaboloid)


#endif /* GATEPARABOLOID_H */
