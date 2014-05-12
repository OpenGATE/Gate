/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDistributionGauss_h
#define GateDistributionGauss_h 1

#include "GateVDistribution.hh"
#include "GateDistributionGaussMessenger.hh"

class GateDistributionGaussMessenger;
class GateDistributionGauss : public GateVDistribution
{
  public:

    //! Constructor
    GateDistributionGauss(const G4String& itsName);
    //! Destructor
    virtual ~GateDistributionGauss() ;

    //! Setters
    inline void SetMean(G4double mean) {m_Mean=mean;}
    inline void SetSigma(G4double sigma) {m_Sigma=sigma;}
    inline void SetAmplitude(G4double amplitude) {m_Amplitude=amplitude;}
    //! Getters
    inline G4double GetMean() const {return m_Mean;}
    inline G4double GetSigma() const {return m_Sigma;}
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
    G4double m_Mean;
    G4double m_Sigma;
    G4double m_Amplitude;
    GateDistributionGaussMessenger* m_messenger;
};


#endif
