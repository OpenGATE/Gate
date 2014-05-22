/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVDistribution_h
#define GateVDistribution_h 1

#include "globals.hh"
#include "GateNamedObject.hh"

/*! \class GateVDistribution
    \brief A generic one variable distribution

    GateVDistribution - by dguez@cea.fr
    This class is the basic class for any distribution description
    All distribution classes must be able to provide
	- a definition domain [ MinX() ; MaxX() ]
	- an image range [ MinY() ; MaxY() ]
	- the value @ each point of the definition domain : GetValue()
	- shoot a random value according to the described distribution : ShootRandom()
	  in case of non-normalized distribution, the shoot must
	  be done following the normalized version....

*/

class GateVDistribution : public GateNamedObject
{
  public:

    //! Constructor
    GateVDistribution(const G4String& itsName);
    //! Destructor
    virtual ~GateVDistribution() ;
    virtual G4double MinX() const=0;
    virtual G4double MinY() const=0;
    virtual G4double MaxX() const=0;
    virtual G4double MaxY() const=0;
    virtual G4double Value(G4double x) const=0;
    // Returns a random number following the current distribution
    // should be optimised according to each distrbution type
    virtual G4double ShootRandom() const=0;

  private:
};


#endif
