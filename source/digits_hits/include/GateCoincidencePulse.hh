/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidencePulse_h
#define GateCoincidencePulse_h 1

#include "GatePulse.hh"
#include "GateCrystalHit.hh"

// define the minimum offset for a delayed coincidence window in sec
#define  MIN_COINC_OFFSET 500.0E-09

class GateCoincidencePulse : public GatePulseList
{
  public:
    inline GateCoincidencePulse(const G4String& itsName,
                                G4double itsCoincidenceWindow,
                                G4double itsOffsetWindow)
      : GatePulseList(itsName),
        m_startTime(DBL_MAX),
//      m_Time(-1),
        m_offsetWindow(itsOffsetWindow),
        m_coincidenceWindow(itsCoincidenceWindow)
    {}
    GateCoincidencePulse(const GateCoincidencePulse& src);
    virtual ~GateCoincidencePulse(){}

    virtual void push_back(GatePulse* newPulse) ;
    virtual void InsertUniqueSortedCopy(GatePulse* newPulse);

    inline G4double GetStartTime() const
      { return m_startTime; }
    inline void SetStartTime(G4double val)
      { m_startTime = val;}
    inline G4double GetTime() const
      { /*return m_Time;*/ return m_startTime+m_offsetWindow+m_coincidenceWindow; }
//     inline void SetTime(G4double val)
//       { m_Time = val;}
    inline G4double GetOffset() const
      { return m_offsetWindow;}
    inline void SetOffset(G4double val)
      { m_offsetWindow = val;}

    virtual G4bool IsInCoincidence(const GatePulse* newPulse) const;
    virtual G4bool IsAfterWindow(const GatePulse* newPulse) const;

    inline G4double GetWindow() const
      { return m_coincidenceWindow;}
    inline void SetWindow(G4double val)
      { m_coincidenceWindow = val;}

    //! Return the min-time of all pulses
    //virtual G4double ComputeStartTime() const ;

    //
    //printing methods
    //
    friend std::ostream& operator<<(std::ostream&, const GateCoincidencePulse&);

  private:
    G4double m_startTime;
//  G4double m_Time;
    G4double m_offsetWindow;
    G4double m_coincidenceWindow;
};

#endif
