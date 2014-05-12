/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDistributionFlat_h
#define GateDistributionFlat_h 1

#include "GateVDistribution.hh"
#include "GateDistributionFlatMessenger.hh"

class GateDistributionFlatMessenger;
class GateDistributionFlat : public GateVDistribution
{
  public:

    //! Constructor
    GateDistributionFlat(const G4String& itsName);
    //! Destructor
    virtual ~GateDistributionFlat() ;

    //! Setters
    inline void SetMin(G4double min) {m_Min=min;}
    inline void SetMax(G4double max) {m_Max=max;}
    inline void SetAmplitude(G4double amplitude) {m_Amplitude=amplitude;}
    //! Getters
    inline G4double GetAmplitude() const {return m_Amplitude;}
    virtual void DescribeMyself(size_t indent);



    virtual G4double MinX() const;
    virtual G4double MinY() const;
    virtual G4double MaxX() const;
    virtual G4double MaxY() const;
    virtual G4double Value(G4double x) const;
    // Returns a random number following the current distribution
    // should be optimised according to each distrbution type
    virtual G4double ShootRandom() const;

  private:
    G4double m_Min;
    G4double m_Max;
    G4double m_Amplitude;
    GateDistributionFlatMessenger* m_messenger;
};


#endif
