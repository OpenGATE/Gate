




/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateDistributionTruncatedGaussian_h
#define GateDistributionTruncatedGaussian_h 1

#include "GateVDistribution.hh"
//#include "GateDistributionTruncatedGaussianMessenger.hh"


class GateDistributionTruncatedGaussian : public GateVDistribution
{
  public:

    //! Constructors
    GateDistributionTruncatedGaussian(const G4String& itsName);
    GateDistributionTruncatedGaussian(const G4String& itsName, 
                                          G4double mu, 
                                          G4double sigma, 
                                          G4double lowLimit, 
                                          G4double highLimit);
    //! Destructor
    virtual ~GateDistributionTruncatedGaussian() ;
   
    //Static methods
    
    //! Setters
    inline void SetMu(G4double mu) {m_Mu=mu;}
    inline void SetSigma(G4double sigma) {m_Sigma=sigma;}
    inline void SetAmplitude(G4double amplitude) 	{m_Amplitude=amplitude;}
    inline void SetLowLimit(G4double lowLimit)		{m_lowLimit = lowLimit;}
    inline void SethighLimit(G4double highLimit)	{m_highLimit = highLimit;}
    //! Getters
    inline G4double GetMu() const {return m_Mu;}
    inline G4double GetSigma() const {return m_Sigma;}
    inline G4double GetAmplitude() const {return m_Amplitude;}
    virtual void DescribeMyself(size_t indent);
    
    

    virtual G4double MinX() const;
    virtual G4double MinY() const;
    virtual G4double MaxX() const;
    virtual G4double MaxY() const;
    virtual G4double Value(G4double x) const;

    //! Math functions
    virtual G4double computeTruncatedSigma() const;
    virtual G4double shootTruncatedGaussian() const;

    //! Static methods;
    static G4double pdf(double x);
    static G4double cdf(double x);
    static G4double shootRandom(G4double mu, G4double sigma, G4double lowLimit, G4double highLimit);   
    static G4double computeTruncatedSigmaStatic(G4double mu, G4double sigma, G4double lowLimit, G4double highLimit) ;

  private:
    G4double m_Mu;
    G4double m_Sigma;
    G4double m_lowLimit;
    G4double m_highLimit;
    G4double m_Amplitude;


};


#endif
