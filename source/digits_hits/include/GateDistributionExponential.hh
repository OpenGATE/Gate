/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDistributionExponential_h
#define GateDistributionExponential_h 1

#include "GateVDistribution.hh"
#include "GateDistributionExponentialMessenger.hh"

class GateDistributionExponentialMessenger;
class GateDistributionExponential : public GateVDistribution
{
  public:

    //! Constructor
    GateDistributionExponential(const G4String& itsName);
    //! Destructor
    virtual ~GateDistributionExponential() ;

    //! Setters
    inline void SetLambda(G4double Lambda) {m_Lambda=Lambda;}
    inline void SetAmplitude(G4double amplitude) {m_Amplitude=amplitude;}
    //! Getters
    inline G4double GetLambda() const {return m_Lambda;}
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
    G4double m_Lambda;
    G4double m_Amplitude;
    GateDistributionExponentialMessenger* m_messenger;
};


#endif
